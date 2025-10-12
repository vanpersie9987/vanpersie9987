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
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@SuppressWarnings("unchecked")
public class Leetcode_3 {
    public static void main(String[] args) {
        // int[] res = numsSameConsecDiff(3, 7);
        // int count = numSteps("1101");
        // int[][] input = { { 1, 1, 1, 1 }, { 1, 1, 2, 1 }, { 1, 2, 1, 1 }, { 1, 1, 1,
        // 1 } };
        // boolean ans = isPrintable(input)
        // int[][] example = { { 0, 2, 0, 0, 1 }, { 0, 2, 0, 2, 2 }, { 0, 2, 0, 0, 0 },
        // { 0, 0, 2, 2, 0 },
        // { 0, 0, 0, 0, 0 } };
        // int res = maximumMinutes(example);
        // int[][] exp = { { 1, 0, 1 }, { 0, 0, 0 }, { 1, 0, 1 } };
        // int res = maxDistance222(exp);
        // int[][] arr = { { 1, 5 }, { 10, 11 }, { 12, 18 }, { 20, 25 }, { 30, 32 } };
        // int res = maximumWhiteTiles(arr, 10);

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
    // 剑指 Offer 55 - I. 二叉树的深度
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int level = 0;
        while (!queue.isEmpty()) {
            ++level;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }
        return level;
    }

    // 104. 二叉树的最大深度 (Maximum Depth of Binary Tree) --dfs
    // 剑指 Offer 55 - I. 二叉树的深度
    public int maxDepth2(TreeNode root) {
        return dfs104(root, 0);
    }

    private int dfs104(TreeNode root, int level) {
        if (root == null) {
            return level;
        }
        return Math.max(dfs104(root.left, level + 1), dfs104(root.right, level + 1));
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

    // 103. 二叉树的锯齿形层序遍历 (Binary Tree Zigzag Level Order Traversal) --bfs
    // 剑指 Offer 32 - III. 从上到下打印二叉树 III --bfs
    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new LinkedList<>();
        if (root == null) {
            return res;
        }
        int level = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> list = new LinkedList<>();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                list.add((level & 1) == 0 ? list.size() : 0, node.val);
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            res.add(list);
            ++level;
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
        return dfs1367(root, head) || isSubPath(head, root.left) || isSubPath(head, root.right);
    }

    private boolean dfs1367(TreeNode treeNode, ListNode listNode) {
        if (listNode == null) {
            return true;
        }
        if (treeNode == null || treeNode.val != listNode.val) {
            return false;
        }
        return dfs1367(treeNode.left, listNode.next) || dfs1367(treeNode.right, listNode.next);
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

    // 23. 合并K个升序链表 (Merge k Sorted Lists) --优先队列
    // 剑指 Offer II 078. 合并排序链表
    public ListNode mergeKLists3(ListNode[] lists) {
        Queue<ListNode> queue = new PriorityQueue<>(new Comparator<ListNode>() {

            @Override
            public int compare(ListNode o1, ListNode o2) {
                return Integer.compare(o1.val, o2.val);
            }

        });

        for (ListNode list : lists) {
            if (list != null) {
                queue.offer(list);
            }

        }

        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;

        while (!queue.isEmpty()) {
            ListNode head = queue.poll();
            cur.next = head;
            if (head.next != null) {
                queue.offer(head.next);
            }
            cur = cur.next;
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
        Integer[] idx = IntStream.range(0, nums.length).boxed().toArray(Integer[]::new);
        Arrays.sort(idx, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(nums[o2], nums[o1]);
            }

        });

        Arrays.sort(idx, 0, k);
        int[] res = new int[k];
        for (int i = 0; i < k; ++i) {
            res[i] = nums[idx[i]];
        }
        return res;

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
    private int m1219;
    private int n1219;
    private int res1219;
    private int[][] grid1219;
    private final int[][] dirs1219 = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

    public int getMaximumGold(int[][] grid) {
        this.m1219 = grid.length;
        this.n1219 = grid[0].length;
        this.grid1219 = grid;
        for (int i = 0; i < m1219; ++i) {
            for (int j = 0; j < n1219; ++j) {
                dfs1219(i, j, grid[i][j]);
            }
        }
        return res1219;

    }

    private void dfs1219(int i, int j, int sum) {
        res1219 = Math.max(res1219, sum);
        int tmp = grid1219[i][j];
        grid1219[i][j] = 0;
        for (int[] d : dirs1219) {
            int ni = i + d[0];
            int nj = j + d[1];
            if (ni >= 0 && ni < m1219 && nj >= 0 && nj < n1219 && grid1219[ni][nj] != 0) {
                dfs1219(ni, nj, sum + grid1219[ni][nj]);
            }
        }
        grid1219[i][j] = tmp;
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
        if ((finalSum & 1) != 0) {
            return List.of();
        }
        List<Long> res = new ArrayList<>();
        long cur = 2L;
        while (cur <= finalSum) {
            res.add(cur);
            finalSum -= cur;
            cur += 2L;
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

    // 2180. 统计各位数字之和为偶数的整数个数 (Count Integers With Even Digit Sum)
    public int countEven2(int num) {
        int y = num / 10;
        int x = num % 10;
        int res = y * 5;
        int t = 0;
        while (y != 0) {
            t += y % 10;
            y /= 10;
        }
        if (t % 2 == 0) {
            res += x / 2 + 1;
        } else {
            res += (x + 1) / 2;
        }
        return res - 1;

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
        int[] cnts = new int[26];
        for (char c : s.toCharArray()) {
            ++cnts[c - 'a'];
        }
        StringBuilder res = new StringBuilder();
        for (int i = 25; i >= 0; --i) {
            search: while (cnts[i] > 0) {
                int max = Math.min(repeatLimit, cnts[i]);
                for (int j = 0; j < max; ++j) {
                    res.append((char) (i + 'a'));
                }
                cnts[i] -= max;
                if (cnts[i] == 0) {
                    break;
                }
                for (int j = i - 1; j >= 0; --j) {
                    if (cnts[j] > 0) {
                        --cnts[j];
                        res.append((char) (j + 'a'));
                        continue search;
                    }
                }
                break;
            }
        }
        return res.toString();

    }

    // 199. 二叉树的右视图 (Binary Tree Right Side View) --bfs
    // 剑指 Offer II 046. 二叉树的右侧视图 --bfs
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
    // 剑指 Offer II 046. 二叉树的右侧视图 --dfs
    private List<Integer> res199;
    private Set<Integer> s199;

    public List<Integer> rightSideView2(TreeNode root) {
        res199 = new ArrayList<>();
        s199 = new HashSet<>();
        dfs199(root, 0);
        return res199;
    }

    private void dfs199(TreeNode root, int d) {
        if (root == null) {
            return;
        }
        if (s199.add(d)) {
            res199.add(root.val);
        }
        dfs199(root.right, d + 1);
        dfs199(root.left, d + 1);
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

    // 965. 单值二叉树 (Univalued Binary Tree) --bfs
    public boolean isUnivalTree3(TreeNode root) {
        int uniVal = root.val;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.val != uniVal) {
                return false;
            }
            if (node.left != null) {
                queue.offer(node.left);
            }
            if (node.right != null) {
                queue.offer(node.right);
            }
        }
        return true;

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
        int n = grid.length;
        Queue<int[]> queue = new LinkedList<>();
        boolean[][] visited = new boolean[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    queue.offer(new int[] { i, j });
                    visited[i][j] = true;
                }
            }
        }
        if (queue.size() == 0 || queue.size() == n * n) {
            return -1;
        }
        int[][] directions = { { 0, -1 }, { 0, 1 }, { -1, 0 }, { 1, 0 } };
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n && grid[nx][ny] == 0) {
                    grid[nx][ny] = grid[cur[0]][cur[1]] + 1;
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                res = Math.max(res, grid[i][j]);
            }
        }
        return res - 1;

    }

    // 542. 01 矩阵 (01 Matrix) --bfs
    // 剑指 Offer II 107. 矩阵中的距离 --bfs
    public int[][] updateMatrix(int[][] mat) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = mat.length;
        int n = mat[0].length;
        boolean[][] seen = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (mat[i][j] == 0) {
                    queue.offer(new int[] { i, j });
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
                    mat[nx][ny] = mat[cur[0]][cur[1]] + 1;
                    queue.offer(new int[] { nx, ny });
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
    private int maxDepth513 = -1;

    public int findBottomLeftValue(TreeNode root) {
        dfs513(root, 0);
        return res513;
    }

    private void dfs513(TreeNode root, int depth) {
        if (root == null) {
            return;
        }
        if (depth > maxDepth513) {
            maxDepth513 = depth;
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
    };

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
            if (counts[player - 1][row] == n || counts[player - 1][n + col] == n || counts[player - 1][2 * n] == n
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

    // 2185. 统计包含给定前缀的字符串 (Counting Words With a Given Prefix)
    public int prefixCount(String[] words, String pref) {
        int res = 0;
        for (String word : words) {
            if (word.startsWith(pref)) {
                ++res;
            }
        }
        return res;

    }

    // 2185. 统计包含给定前缀的字符串 (Counting Words With a Given Prefix) --字典树
    public int prefixCount2(String[] words, String pref) {
        Trie2185 trie = new Trie2185();
        for (String word : words) {
            trie.insert(word);
        }
        return trie.getCount(pref);

    }

    public class Trie2185 {
        private Trie2185[] children;
        private int count;

        public Trie2185() {
            children = new Trie2185[26];
        }

        public void insert(String s) {
            Trie2185 trie = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (trie.children[index] == null) {
                    trie.children[index] = new Trie2185();
                }
                trie = trie.children[index];
                ++trie.count;
            }
        }

        public int getCount(String s) {
            Trie2185 trie = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (trie.children[index] == null) {
                    return 0;
                }
                trie = trie.children[index];
            }
            return trie.count;
        }

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

    // 2187. 完成旅途的最少时间 (Minimum Time to Complete Trips)
    public long minimumTime(int[] time, int totalTrips) {
        long left = 1L;
        long right = (long) Arrays.stream(time).max().getAsInt() * totalTrips;
        long res = 1L;
        while (left <= right) {
            long mid = left + ((right - left) >> 1);
            if (check2187(mid, time, totalTrips)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private boolean check2187(long t, int[] time, int totalTrips) {
        long s = 0;
        for (int c : time) {
            s += t / c;
            if (s >= totalTrips) {
                return true;
            }
        }
        return false;
    }

    // 346. 数据流中的移动平均值 (Moving Average from Data Stream) --plus
    // 剑指 Offer II 041. 滑动窗口的平均值
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

    // 244. 最短单词距离 II (Shortest Word Distance II)
    class WordDistance {
        private Map<String, List<Integer>> map;

        public WordDistance(String[] wordsDict) {
            map = new HashMap<>();
            for (int i = 0; i < wordsDict.length; ++i) {
                map.computeIfAbsent(wordsDict[i], k -> new ArrayList<>()).add(i);
            }

        }

        public int shortest(String word1, String word2) {
            List<Integer> list1 = map.get(word1);
            List<Integer> list2 = map.get(word2);
            int i = 0;
            int j = 0;
            int res = Integer.MAX_VALUE;
            while (i < list1.size() && j < list2.size()) {
                int index1 = list1.get(i);
                int index2 = list2.get(j);
                res = Math.min(res, Math.abs(index1 - index2));
                if (index1 < index2) {
                    ++i;
                } else {
                    ++j;
                }
            }
            return res;
        }
    }

    // 245. 最短单词距离 III (Shortest Word Distance III)
    public int shortestWordDistance(String[] wordsDict, String word1, String word2) {
        int res = Integer.MAX_VALUE;
        if (word1.equals(word2)) {
            int pre = -1;
            for (int i = 0; i < wordsDict.length; ++i) {
                if (word1.equals(wordsDict[i])) {
                    if (pre != -1) {
                        res = Math.min(res, i - pre);
                    }
                    pre = i;
                }
            }
        } else {
            int index1 = -1;
            int index2 = -1;
            for (int i = 0; i < wordsDict.length; ++i) {
                if (word1.equals(wordsDict[i])) {
                    index1 = i;
                } else if (word2.equals(wordsDict[i])) {
                    index2 = i;
                }
                if (index1 != -1 && index2 != -1) {
                    res = Math.min(res, Math.abs(index1 - index2));
                }
            }
        }
        return res;

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

    // 573. 松鼠模拟 (Squirrel Simulation) --plus
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
        long res = 0L;
        int n = nums.length;
        int pre = 0;
        for (int num : nums) {
            if (num - pre <= 1) {
                pre = num;
                continue;
            }
            int first = pre + 1;
            int last = Math.min(first + k - 1, num - 1);
            res += (long) (first + last) * (last - first + 1) / 2;
            pre = num;
            k -= last - first + 1;
            if (k <= 0) {
                break;
            }
        }
        int first = nums[n - 1] + 1;
        int last = first + k - 1;
        res += (long) (first + last) * (last - first + 1) / 2;
        return res;

    }

    // 2196. 根据描述创建二叉树 (Create Binary Tree From Descriptions)
    private Map<Integer, List<int[]>> g2196;

    public TreeNode createBinaryTree(int[][] descriptions) {
        Set<Integer> set = new HashSet<>();
        this.g2196 = new HashMap<>();
        for (int[] d : descriptions) {
            g2196.computeIfAbsent(d[0], k -> new ArrayList<>()).add(new int[] { d[1], d[2] });
            set.add(d[1]);
        }
        int r = -1;
        for (int[] d : descriptions) {
            if (!set.contains(d[0])) {
                r = d[0];
                break;
            }
        }
        TreeNode root = new TreeNode(r);
        dfs2196(root);
        return root;

    }

    private void dfs2196(TreeNode root) {
        int x = root.val;
        for (int[] son : g2196.getOrDefault(x, new ArrayList<>())) {
            int isLeft = son[1];
            int y = son[0];
            TreeNode node = new TreeNode(y);
            if (isLeft == 1) {
                root.left = node;
            } else {
                root.right = node;
            }
            dfs2196(node);
        }
    }

    // 733. 图像渲染 (Flood Fill) --bfs
    // 面试题 08.10. 颜色填充 (Color Fill LCCI)
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor) {
        int m = image.length;
        int n = image[0].length;
        if (image[sr][sc] == newColor) {
            return image;
        }
        int originalColor = image[sr][sc];
        int[][] directions = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { sr, sc });
        image[sr][sc] = newColor;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && image[nx][ny] == originalColor) {
                    image[nx][ny] = newColor;
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        return image;
    }

    // 面试题 04.01. 节点间通路 (Route Between Nodes LCCI) --bfs
    public boolean findWhetherExistsPath(int n, int[][] graph, int start, int target) {
        Map<Integer, Set<Integer>> map = new HashMap<>();
        boolean[] visited = new boolean[n];
        for (int[] g : graph) {
            // 过滤自环
            if (g[0] == g[1]) {
                continue;
            }
            // 使用set，过滤重边
            map.computeIfAbsent(g[0], k -> new HashSet<>()).add(g[1]);
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
            for (int neighbor : map.get(key)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.offer(neighbor);
                }
            }
        }
        return false;
    }

    // 面试题 04.01. 节点间通路 (Route Between Nodes LCCI)
    private int target04_01;
    private List<Integer>[] g04_01;
    private boolean[] vis04_01;

    public boolean findWhetherExistsPath2(int n, int[][] graph, int start, int target) {
        this.g04_01 = new ArrayList[n];
        Arrays.setAll(g04_01, k -> new ArrayList<>());
        for (int[] gr : graph) {
            g04_01[gr[0]].add(gr[1]);
        }
        this.target04_01 = target;
        this.vis04_01 = new boolean[n];
        return dfs04_01(start);

    }

    private boolean dfs04_01(int x) {
        if (x == target04_01) {
            return true;
        }
        if (vis04_01[x]) {
            return false;
        }
        vis04_01[x] = true;
        for (int y : g04_01[x]) {
            if (dfs04_01(y)) {
                return true;
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

    // 1379. 找出克隆二叉树中的相同节点 (Find a Corresponding Node of a Binary Tree in a Clone of
    // That Tree)
    public final TreeNode getTargetCopy2(final TreeNode original, final TreeNode cloned, final TreeNode target) {
        return dfs1379(original, cloned, target);
    }

    private TreeNode dfs1379(TreeNode x, TreeNode y, TreeNode target) {
        if (x == null) {
            return null;
        }
        if (x == target) {
            return y;
        }
        TreeNode res = dfs1379(x.left, y.left, target);
        if (res != null) {
            return res;
        }
        return dfs1379(x.right, y.right, target);
    }

    // 515. 在每个树行中找最大值 (Find Largest Value in Each Tree Row) --bfs
    // 剑指 Offer II 044. 二叉树每层的最大值
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

    // 1992. 找到所有的农场组 (Find All Groups of Farmland) --bfs
    public int[][] findFarmland2(int[][] land) {
        int[][] directions = { { 0, 1 }, { 1, 0 } };
        List<int[]> res = new ArrayList<>();
        int m = land.length;
        int n = land[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (land[i][j] == 1) {
                    land[i][j] = 0;
                    Queue<int[]> queue = new LinkedList<>();
                    int endX = i;
                    int endY = j;
                    queue.offer(new int[] { i, j });
                    while (!queue.isEmpty()) {
                        int[] cur = queue.poll();
                        endX = cur[0];
                        endY = cur[1];
                        for (int[] direction : directions) {
                            int nx = endX + direction[0];
                            int ny = endY + direction[1];
                            if (nx >= 0 && nx < m && ny >= 0 && ny < n && land[nx][ny] == 1) {
                                land[nx][ny] = 0;
                                queue.offer(new int[] { nx, ny });
                            }
                        }
                    }
                    res.add(new int[] { i, j, endX, endY });
                }
            }
        }
        return res.toArray(new int[0][]);

    }

    // 417. 太平洋大西洋水流问题 (Pacific Atlantic Water Flow) --bfs
    public List<List<Integer>> pacificAtlantic(int[][] heights) {
        int m = heights.length;
        int n = heights[0].length;
        boolean[][] vis1 = bfs417(heights, 0, 0);
        boolean[][] vis2 = bfs417(heights, m - 1, n - 1);
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (vis1[i][j] && vis2[i][j]) {
                    res.add(List.of(i, j));
                }
            }
        }
        return res;

    }

    private boolean[][] bfs417(int[][] heights, int r, int c) {
        int m = heights.length;
        int n = heights[0].length;
        Queue<int[]> q = new ArrayDeque<>();
        boolean[][] vis = new boolean[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == r || j == c) {
                    vis[i][j] = true;
                    q.offer(new int[] { i, j });
                }
            }
        }
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        while (!q.isEmpty()) {
            int[] p = q.poll();
            int x = p[0];
            int y = p[1];
            for (int[] d : dirs) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !vis[nx][ny] && heights[nx][ny] >= heights[x][y]) {
                    vis[nx][ny] = true;
                    q.offer(new int[] { nx, ny });
                }
            }
        }
        return vis;

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

    // 127. 单词接龙 (Word Ladder) --bfs 还需掌握 双向bfs、优化建图
    // 剑指 Offer II 108. 单词演变
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        if (!wordList.contains(endWord)) {
            return 0;
        }
        int n = wordList.size();
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; ++i) {
            if (wordList.get(i).equals(beginWord)) {
                visited[i] = true;
                break;
            }
        }
        Queue<String> queue = new LinkedList<>();
        queue.offer(beginWord);
        int res = 1;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                String cur = queue.poll();
                if (cur.equals(endWord)) {
                    return res;
                }
                for (int j = 0; j < n; ++j) {
                    if (!visited[j] && differByOnlyOneCharacter(cur, wordList.get(j))) {
                        visited[j] = true;
                        queue.offer(wordList.get(j));
                    }
                }
            }
            if (queue.isEmpty()) {
                return 0;
            }
            ++res;
        }
        return 0;

    }

    private boolean differByOnlyOneCharacter(String s1, String s2) {
        int diff = 0;
        int n = s1.length();
        for (int i = 0; i < n; ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                if (++diff > 1) {
                    return false;
                }
            }
        }
        return diff == 1;
    }

    // 752. 打开转盘锁 (Open the Lock) --bfs 还需掌握 启发式搜索
    // 剑指 Offer II 109. 开密码锁 --bfs
    public int openLock(String[] deadends, String target) {
        if (target.equals("0000")) {
            return 0;
        }
        Set<String> set = new HashSet<>();
        for (String deadend : deadends) {
            set.add(deadend);
        }
        if (set.contains("0000")) {
            return -1;
        }
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.offer("0000");
        visited.add("0000");
        int res = 0;
        while (!queue.isEmpty()) {
            ++res;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                char[] cur = queue.poll().toCharArray();
                for (int j = 0; j < 4; ++j) {
                    char temp = cur[j];
                    int bit = cur[j] - '0';
                    char change1 = (char) ((bit + 1) % 10 + '0');
                    cur[j] = change1;
                    String candidate1 = String.valueOf(cur);
                    if (target.equals(candidate1)) {
                        return res;
                    }
                    if (!set.contains(candidate1) && !visited.contains(candidate1)) {
                        visited.add(candidate1);
                        queue.offer(candidate1);
                    }
                    char change2 = (char) ((bit - 1 + 10) % 10 + '0');
                    cur[j] = change2;
                    String candidate2 = String.valueOf(cur);
                    if (target.equals(candidate2)) {
                        return res;
                    }
                    if (!set.contains(candidate2) && !visited.contains(candidate2)) {
                        visited.add(candidate2);
                        queue.offer(candidate2);
                    }
                    cur[j] = temp;
                }
            }
        }
        return -1;

    }

    // 909. 蛇梯棋 (Snakes and Ladders) --bfs
    public int snakesAndLadders(int[][] board) {
        int n = board.length;
        int[] newBoard = new int[n * n];
        int idx = 0;
        for (int i = n - 1; i >= 0; --i) {
            if ((n - 1 - i) % 2 == 0) {
                for (int j = 0; j < n; ++j) {
                    newBoard[idx++] = board[i][j] - 1;
                }
            } else {
                for (int j = n - 1; j >= 0; --j) {
                    newBoard[idx++] = board[i][j] - 1;
                }
            }
        }
        boolean[] vis = new boolean[n * n];
        vis[0] = true;
        Queue<Integer> q = new LinkedList<>();
        q.offer(0);
        int res = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int x = q.poll();
                if (x == n * n - 1) {
                    return res;
                }
                for (int j = x + 1; j <= Math.min(x + 6, n * n - 1); ++j) {
                    if (newBoard[j] < 0 && !vis[j]) {
                        vis[j] = true;
                        q.offer(j);
                    } else if (newBoard[j] >= 0 && !vis[newBoard[j]]) {
                        int next = newBoard[j];
                        vis[next] = true;
                        q.offer(next);
                    }
                }
            }
            ++res;
        }
        return -1;

    }

    // 994. 腐烂的橘子 (Rotting Oranges) --bfs
    public int orangesRotting(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int rotten = 0;
        int fresh = 0;
        Queue<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    ++fresh;
                } else if (grid[i][j] == 2) {
                    ++rotten;
                    q.offer(new int[] { i, j });
                }
            }
        }
        if (fresh == 0) {
            return 0;
        }
        if (rotten == 0) {
            return -1;
        }
        int res = 0;
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int[] p = q.poll();
                int x = p[0];
                int y = p[1];
                for (int[] d : dirs) {
                    int nx = x + d[0];
                    int ny = y + d[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                        grid[nx][ny] = 2;
                        --fresh;
                        q.offer(new int[] { nx, ny });
                    }
                }
            }
            if (!q.isEmpty()) {
                ++res;
            }
        }
        if (fresh > 0) {
            return -1;
        }
        return res;

    }

    // 433. 最小基因变化 (Minimum Genetic Mutation) --dfs
    public int minMutation(String start, String end, String[] bank) {
        Set<String> set = new HashSet<>();
        for (String item : bank) {
            set.add(item);
        }
        if (!set.contains(end)) {
            return -1;
        }
        set.clear();
        int res = 0;
        Queue<String> queue = new LinkedList<>();
        queue.offer(start);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                String cur = queue.poll();
                if (cur.equals(end)) {
                    return res;
                }
                for (String item : bank) {
                    if (!set.contains(item) && isDiffOnlyOne(cur, item)) {
                        set.add(item);
                        queue.offer(item);
                    }
                }
            }
            ++res;
        }
        return -1;

    }

    private boolean isDiffOnlyOne(String word1, String word2) {
        int diff = 0;
        for (int i = 0; i < word1.length(); ++i) {
            if (word1.charAt(i) != word2.charAt(i)) {
                if (++diff > 1) {
                    return false;
                }
            }
        }
        return diff == 1;
    }

    // 429. N 叉树的层序遍历 (N-ary Tree Level Order Traversal) --bfs
    public List<List<Integer>> levelOrder(Node root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < size; ++i) {
                Node node = queue.poll();
                list.add(node.val);
                for (Node child : node.children) {
                    queue.offer(child);
                }
            }
            res.add(list);
        }
        return res;

    }

    // 429. N 叉树的层序遍历 (N-ary Tree Level Order Traversal)
    private List<List<Integer>> res429;

    public List<List<Integer>> levelOrder2(Node root) {
        this.res429 = new ArrayList<>();
        dfs429(root, 0);
        return res429;

    }

    private void dfs429(Node root, int d) {
        if (root == null) {
            return;
        }
        if (res429.size() == d) {
            res429.add(new ArrayList<>());
        }
        res429.get(d).add(root.val);
        for (Node y : root.children) {
            dfs429(y, d + 1);
        }
    }

    // 559. N 叉树的最大深度 (Maximum Depth of N-ary Tree)
    public int maxDepth(Node root) {
        if (root == null) {
            return 0;
        }
        int max = 1;
        for (Node y : root.children) {
            max = Math.max(max, maxDepth(y) + 1);
        }
        return max;
    }

    // 559. N 叉树的最大深度 (Maximum Depth of N-ary Tree) --bfs
    public int maxDepth2(Node root) {
        int res = 0;
        if (root == null) {
            return res;
        }
        Queue<Node> queue = new LinkedList<>();
        queue.offer(root);
        res = 0;
        while (!queue.isEmpty()) {
            ++res;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                Node cur = queue.poll();
                for (Node child : cur.children) {
                    queue.offer(child);
                }
            }
        }
        return res;
    }

    // 841. 钥匙和房间 (Keys and Rooms) --bfs
    public boolean canVisitAllRooms(List<List<Integer>> rooms) {
        int n = rooms.size();
        boolean[] visited = new boolean[n];
        Queue<Integer> queue = new LinkedList<>();
        visited[0] = true;
        queue.offer(0);
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            for (int i : rooms.get(cur)) {
                if (!visited[i]) {
                    queue.offer(i);
                    visited[i] = true;
                }
            }
        }
        for (boolean visit : visited) {
            if (!visit) {
                return false;
            }
        }
        return true;
    }

    // 797. 所有可能的路径 (All Paths From Source to Target) --bfs
    // 剑指 Offer II 110. 所有路径
    public List<List<Integer>> allPathsSourceTarget(int[][] graph) {
        int n = graph.length;
        List<List<Integer>> res = new ArrayList<>();
        Queue<List<Integer>> queue = new LinkedList<>();
        for (int node : graph[0]) {
            List<Integer> list = new ArrayList<>();
            list.add(0);
            list.add(node);
            queue.offer(list);
        }
        while (!queue.isEmpty()) {
            List<Integer> cur = queue.poll();
            int lastNode = cur.get(cur.size() - 1);
            if (lastNode == n - 1) {
                res.add(cur);
            } else {
                for (int neighbor : graph[lastNode]) {
                    List<Integer> nList = new ArrayList<>(cur);
                    nList.add(neighbor);
                    queue.offer(nList);
                }
            }
        }
        return res;

    }

    // 797. 所有可能的路径 (All Paths From Source to Target)
    // 剑指 Offer II 110. 所有路径
    private List<Integer>[] g797;
    private int n797;
    private List<List<Integer>> res797;

    public List<List<Integer>> allPathsSourceTarget2(int[][] graph) {
        this.n797 = graph.length;
        this.g797 = new ArrayList[n797];
        Arrays.setAll(g797, k -> new ArrayList<>());
        for (int i = 0; i < n797; ++i) {
            for (int j : graph[i]) {
                g797[i].add(j);
            }
        }
        this.res797 = new ArrayList<>();
        dfs797(0, new ArrayList<>());
        return res797;

    }

    private void dfs797(int x, List<Integer> list) {
        list.add(x);
        if (x == n797 - 1) {
            res797.add(new ArrayList<>(list));
            return;
        }
        for (int y : g797[x]) {
            dfs797(y, list);
            list.remove(list.size() - 1);
        }
    }

    // 2044. 统计按位或能得到最大值的子集数目 (Count Number of Maximum Bitwise-OR Subsets)
    public int countMaxOrSubsets(int[] nums) {
        int n = nums.length;
        int max = 0;
        int res = 0;
        int[] s = new int[1 << n];
        for (int i = 1; i < 1 << n; ++i) {
            int lb = Integer.numberOfTrailingZeros(i);
            s[i] = s[i ^ (1 << lb)] | nums[lb];
            if (s[i] > max) {
                max = s[i];
                res = 1;
            } else if (s[i] == max) {
                ++res;
            }
        }
        return res;

    }

    // 2044. 统计按位或能得到最大值的子集数目 (Count Number of Maximum Bitwise-OR Subsets)
    private int max2044;
    private int[] nums2044;
    private int n2044;

    public int countMaxOrSubsets2(int[] nums) {
        this.n2044 = nums.length;
        this.nums2044 = nums;
        this.max2044 = 0;
        for (int x : nums) {
            max2044 |= x;
        }
        return dfs2044(0, 0);

    }

    private int dfs2044(int i, int j) {
        if (i == n2044) {
            return j == max2044 ? 1 : 0;
        }
        return dfs2044(i + 1, j) + dfs2044(i + 1, j | nums2044[i]);
    }

    // 1625. 执行操作后字典序最小的字符串 (Lexicographically Smallest String After Applying
    // Operations) --bfs
    public String findLexSmallestString(String s, int a, int b) {
        String res = s;
        Set<String> visited = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        queue.offer(s);
        visited.add(s);
        while (!queue.isEmpty()) {
            String cur = queue.poll();
            if (cur.compareTo(res) < 0) {
                res = cur;
            }
            char[] curChars = cur.toCharArray();
            for (int i = 1; i < curChars.length; i += 2) {
                curChars[i] = (char) (((curChars[i] - '0' + a) % 10) + '0');
            }
            if (visited.add(String.valueOf(curChars))) {
                queue.offer(String.valueOf(curChars));
            }
            cur = cur.substring(cur.length() - b) + cur.substring(0, cur.length() - b);
            if (visited.add(cur)) {
                queue.offer(cur);
            }
        }
        return res;

    }

    // 847. 访问所有节点的最短路径 (Shortest Path Visiting All Nodes) --bfs + 状态压缩
    public int shortestPathLength(int[][] graph) {
        int n = graph.length;
        boolean[][] visited = new boolean[n][1 << n];
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            queue.offer(new int[] { i, 1 << i });
            visited[i][1 << i] = true;
        }
        int res = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = queue.poll();
                int node = cur[0];
                int bitMask = cur[1];
                if (bitMask == ((1 << n) - 1)) {
                    return res;
                }
                for (int neighbor : graph[node]) {
                    int nMask = (1 << neighbor) | bitMask;
                    if (!visited[neighbor][nMask]) {
                        visited[neighbor][nMask] = true;
                        queue.offer(new int[] { neighbor, nMask });
                    }
                }
            }
            ++res;
        }
        return -1;

    }

    // 1034. 边界着色 (Coloring A Border) --bfs
    public int[][] colorBorder(int[][] grid, int row, int col, int color) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        List<int[]> changedList = new ArrayList<>();
        boolean[][] visited = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { row, col });
        visited[row][col] = true;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            boolean flag = false;
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (!(nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == grid[row][col])) {
                    flag = true;
                } else if (!visited[nx][ny]) {
                    visited[nx][ny] = true;
                    queue.offer(new int[] { nx, ny });
                }
            }
            if (flag) {
                changedList.add(new int[] { cur[0], cur[1] });
            }
        }
        for (int[] changColor : changedList) {
            grid[changColor[0]][changColor[1]] = color;
        }
        return grid;

    }

    // 967. 连续差相同的数字 (Numbers With Same Consecutive Differences) --bfs
    public int[] numsSameConsecDiff(int n, int k) {
        List<Integer> candidates = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 1; i <= 9; ++i) {
            queue.offer(i);
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            if (String.valueOf(cur).length() == n) {
                candidates.add(cur);
            } else {
                if (cur % 10 - k >= 0) {
                    queue.offer(cur * 10 + cur % 10 - k);
                }
                if (k != 0 && cur % 10 + k <= 9) {
                    queue.offer(cur * 10 + cur % 10 + k);
                }
            }
        }
        int[] res = new int[candidates.size()];
        for (int i = 0; i < candidates.size(); ++i) {
            res[i] = candidates.get(i);
        }
        return res;

    }

    // 934. 最短的桥 (Shortest Bridge) --bfs
    public int shortestBridge(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        Queue<Integer> queue = new LinkedList<>();
        boolean[][] visited = new boolean[m][n];
        boolean flag = false;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    flag = true;
                    queue.offer(i * n + j);
                    visited[i][j] = true;
                    break;
                }
            }
            if (flag) {
                break;
            }
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            int x = cur / n;
            int y = cur % n;
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] && grid[nx][ny] == 1) {
                    visited[nx][ny] = true;
                    queue.offer(nx * n + ny);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (visited[i][j]) {
                    queue.offer(i * n + j);
                }
            }
        }
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int cur = queue.poll();
                int x = cur / n;
                int y = cur % n;
                for (int[] direction : directions) {
                    int nx = x + direction[0];
                    int ny = y + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]) {
                        if (grid[nx][ny] == 1) {
                            return res;
                        }
                        visited[nx][ny] = true;
                        queue.offer(nx * n + ny);
                    }
                }
            }
            ++res;
        }
        return res;

    }

    // 720. 词典中最长的单词 (Longest Word in Dictionary) --哈希表 + 排序
    public String longestWord(String[] words) {
        Arrays.sort(words, (o1, o2) -> o1.length() != o2.length() ? o1.length() - o2.length() : o2.compareTo(o1));
        String res = "";
        Set<String> set = new HashSet<>();
        set.add("");
        for (int i = 0; i < words.length; ++i) {
            if (set.contains(words[i].substring(0, words[i].length() - 1))) {
                res = words[i];
                set.add(words[i]);
            }
        }
        return res;

    }

    // 1926. 迷宫中离入口最近的出口 (Nearest Exit from Entrance in Maze) --bfs
    public int nearestExit(char[][] maze, int[] entrance) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = maze.length;
        int n = maze[0].length;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(entrance);
        maze[entrance[0]][entrance[1]] = '+';
        int res = 0;
        while (!queue.isEmpty()) {
            ++res;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = queue.poll();
                for (int[] direction : directions) {
                    int nx = cur[0] + direction[0];
                    int ny = cur[1] + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && maze[nx][ny] == '.') {
                        if (nx == 0 || nx == m - 1 || ny == 0 || ny == n - 1) {
                            return res;
                        }
                        maze[nx][ny] = '+';
                        queue.offer(new int[] { nx, ny });
                    }
                }
            }
        }
        return -1;

    }

    // 407. 接雨水 II (Trapping Rain Water II) --Dijkstra
    public int trapRainWater(int[][] heightMap) {
        int m = heightMap.length;
        int n = heightMap[0].length;
        int res = 0;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    q.offer(new int[] { heightMap[i][j], i, j });
                    heightMap[i][j] = -1;
                }

            }
        }
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int h = cur[0];
            int x = cur[1];
            int y = cur[2];
            for (int[] d : dirs) {
                int dx = d[0];
                int dy = d[1];
                int nx = x + dx;
                int ny = y + dy;
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && heightMap[nx][ny] >= 0) {
                    res += Math.max(0, h - heightMap[nx][ny]);
                    q.offer(new int[] { Math.max(heightMap[nx][ny], h), nx, ny });
                    heightMap[nx][ny] = -1;
                }
            }
        }
        return res;

    }

    // 407. 接雨水 II (Trapping Rain Water II) --bfs
    public int trapRainWater2(int[][] heightMap) {
        int maxHeight = 0;
        int m = heightMap.length;
        int n = heightMap[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                maxHeight = Math.max(maxHeight, heightMap[i][j]);
            }
        }
        int[][] water = new int[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                water[i][j] = maxHeight;
            }
        }

        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1) {
                    if (water[i][j] > heightMap[i][j]) {
                        water[i][j] = heightMap[i][j];
                        queue.offer(new int[] { i, j });
                    }
                }
            }
        }
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && water[nx][ny] > water[cur[0]][cur[1]]
                        && water[nx][ny] > heightMap[nx][ny]) {
                    water[nx][ny] = Math.max(water[cur[0]][cur[1]], heightMap[nx][ny]);
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                res += water[i][j] - heightMap[i][j];
            }
        }
        return res;

    }

    // 1315. 祖父节点值为偶数的节点和 (Sum of Nodes with Even-Valued Grandparent) --bfs
    public int sumEvenGrandparent(TreeNode root) {
        int res = 0;
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode cur = queue.poll();
            if (cur.left != null) {
                queue.offer(cur.left);
                if (cur.val % 2 == 0) {
                    if (cur.left.left != null) {
                        res += cur.left.left.val;
                    }
                    if (cur.left.right != null) {
                        res += cur.left.right.val;
                    }
                }
            }
            if (cur.right != null) {
                queue.offer(cur.right);
                if (cur.val % 2 == 0) {
                    if (cur.right.left != null) {
                        res += cur.right.left.val;
                    }
                    if (cur.right.right != null) {
                        res += cur.right.right.val;
                    }
                }
            }
        }
        return res;

    }

    // 637. 二叉树的层平均值 (Average of Levels in Binary Tree) --bfs
    public List<Double> averageOfLevels(TreeNode root) {
        List<Double> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            double sum = 0d;
            for (int i = 0; i < size; ++i) {
                TreeNode cur = queue.poll();
                sum += cur.val;
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            res.add(sum / size);
        }
        return res;

    }

    // 6020. 将数组划分成相等数对 (Divide Array Into Equal Pairs)
    public boolean divideArray(int[] nums) {
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i += 2) {
            if (nums[i] != nums[i + 1]) {
                return false;
            }
        }
        return true;

    }

    // 2207. 字符串中最多数目的子序列 (Maximize Number of Subsequences in a String)
    public long maximumSubsequenceCount(String text, String pattern) {
        return Math.max(check2207(text + pattern.charAt(1), pattern), check2207(pattern.charAt(0) + text, pattern));
    }

    private long check2207(String s, String p) {
        int cnt0 = 0;
        long res = 0L;
        char p0 = p.charAt(0);
        char p1 = p.charAt(1);
        for (char c : s.toCharArray()) {
            if (c == p1) {
                res += cnt0;
            }
            if (c == p0) {
                ++cnt0;
            }
        }
        return res;
    }

    // 6022. 将数组和减半的最少操作次数 (Minimum Operations to Halve Array Sum)
    public int halveArray(int[] nums) {
        PriorityQueue<Double> priorityQueue = new PriorityQueue<>(new Comparator<Double>() {

            @Override
            public int compare(Double o1, Double o2) {
                return o2.compareTo(o1);
            }

        });
        double sum = 0;
        for (int num : nums) {
            sum += num;
            priorityQueue.offer((double) num);
        }
        int res = 0;
        double deleteSum = 0;
        while (deleteSum < sum / 2) {
            double max = priorityQueue.poll();
            max /= 2;
            deleteSum += max;
            priorityQueue.offer(max);
            ++res;
        }
        return res;
    }

    // 2200. 找出数组中的所有 K 近邻下标 (Find All K-Distant Indices in an Array)
    public List<Integer> findKDistantIndices(int[] nums, int key, int k) {
        int n = nums.length;
        int cnt = 0;
        for (int i = 0; i < Math.min(n, k + 1); ++i) {
            if (nums[i] == key) {
                ++cnt;
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            if (cnt > 0) {
                res.add(i);
            }
            if (i + k + 1 < n && nums[i + k + 1] == key) {
                ++cnt;
            }
            if (i - k >= 0 && nums[i - k] == key) {
                --cnt;
            }
        }
        return res;

    }

    // 2201. 统计可以提取的工件 (Count Artifacts That Can Be Extracted)
    public int digArtifacts(int n, int[][] artifacts, int[][] dig) {
        boolean[][] isDig = new boolean[n][n];
        for (int[] d : dig) {
            isDig[d[0]][d[1]] = true;
        }
        int res = 0;
        search: for (int[] artifact : artifacts) {
            int startX = artifact[0];
            int startY = artifact[1];
            int endX = artifact[2];
            int endY = artifact[3];
            for (int i = startX; i <= endX; ++i) {
                for (int j = startY; j <= endY; ++j) {
                    if (!isDig[i][j]) {
                        continue search;
                    }
                }
            }
            ++res;
        }
        return res;

    }

    // 2202. K 次操作后最大化顶端元素 (Maximize the Topmost Element After K Moves)
    public int maximumTop(int[] nums, int k) {
        int n = nums.length;
        if (n == 1) {
            return k % 2 == 1 ? -1 : nums[0];
        }
        if (k <= 1) {
            return nums[k];
        }
        if (k > n) {
            return Arrays.stream(nums).max().getAsInt();
        }
        if (k == n) {
            return Arrays.stream(Arrays.copyOfRange(nums, 0, n - 1)).max().getAsInt();
        }
        // 1 < k < n
        return Math.max(Arrays.stream(Arrays.copyOfRange(nums, 0, k - 1)).max().getAsInt(), nums[k]);

    }

    // 银联-01. 回文链表
    public boolean isPalindrome(ListNode head) {
        List<Integer> list = new ArrayList<>();
        while (head != null) {
            list.add(head.val);
            head = head.next;
        }
        int left = 0;
        int right = list.size() - 1;
        while (left < right) {
            if (list.get(left) == list.get(right)) {
                ++left;
                --right;
            } else {
                return judge01(list, left + 1, right) || judge01(list, left, right - 1);
            }
        }
        return true;

    }

    private boolean judge01(List<Integer> list, int left, int right) {
        while (left < right) {
            if (list.get(left) != list.get(right)) {
                return false;
            }
            ++left;
            --right;
        }
        return true;
    }

    // 面试题 10.10. 数字流的秩 (Rank from Stream LCCI)
    // --还需掌握树状数组、二分查找
    class StreamRank {
        private int[] counts;

        public StreamRank() {
            counts = new int[50001];

        }

        public void track(int x) {
            ++counts[x];
        }

        public int getRankOfNumber(int x) {
            int res = 0;
            for (int i = 0; i <= x; ++i) {
                res += counts[i];
            }
            return res;
        }
    }

    // 453. 最小操作次数使数组元素相等 (Minimum Moves to Equal Array Elements)
    public int minMoves(int[] nums) {
        int min = Integer.MAX_VALUE;
        for (int num : nums) {
            min = Math.min(min, num);
        }
        int res = 0;
        for (int num : nums) {
            res += num - min;
        }
        return res;

    }

    // 2210. 统计数组中峰和谷的数量 (Count Hills and Valleys in an Array)
    public int ountHillValley(int[] nums) {
        List<Integer> a = new ArrayList<>();
        for (int i = 0; i < nums.length;) {
            a.add(nums[i]);
            int j = i;
            while (j < nums.length && nums[j] == nums[i]) {
                ++j;
            }
            i = j;
        }
        int res = 0;
        for (int i = 1; i < a.size() - 1; ++i) {
            if (a.get(i) > a.get(i - 1) && a.get(i) > a.get(i + 1)) {
                ++res;
            } else if (a.get(i) < a.get(i - 1) && a.get(i) < a.get(i + 1)) {
                ++res;
            }
        }
        return res;

    }

    // 2211. 统计道路上的碰撞次数 (Count Collisions on a Road)
    public int countCollisions(String directions) {
        int res = 0;
        char[] chars = directions.toCharArray();
        int i = 0;
        int j = chars.length - 1;
        while (i < chars.length) {
            if (chars[i] != 'L') {
                break;
            }
            ++i;
        }
        while (j >= 0) {
            if (chars[j] != 'R') {
                break;
            }
            --j;
        }
        for (int k = i; k <= j; ++k) {
            if (chars[k] != 'S') {
                ++res;
            }
        }
        return res;

    }

    // 2212. 射箭比赛中的最大得分 (Maximum Points in an Archery Competition)
    public int[] maximumBobPoints(int numArrows, int[] aliceArrows) {
        int max = 0;
        int status = 0;
        for (int i = 0; i < (1 << 12); ++i) {
            int cur = getRes6029(numArrows, aliceArrows, i);
            if (cur > max) {
                max = cur;
                status = i;
            }
        }
        int[] res = new int[12];
        for (int i = 0; i < 12; ++i) {
            if (((status >>> i) & 1) == 1) {
                if (aliceArrows[i] + 1 <= numArrows) {
                    res[i] = aliceArrows[i] + 1;
                    numArrows -= aliceArrows[i] + 1;
                    if (numArrows == 0) {
                        return res;
                    }
                }
            }
        }
        res[0] += numArrows;
        return res;

    }

    private int getRes6029(int numArrows, int[] aliceArrows, int status) {
        int res = 0;
        for (int i = 0; i < 12; ++i) {
            if (((status >>> i) & 1) == 1) {
                if (aliceArrows[i] + 1 <= numArrows) {
                    res += i;
                    numArrows -= aliceArrows[i] + 1;
                    if (numArrows == 0) {
                        return res;
                    }
                }
            }
        }
        return res;
    }

    // 207. 课程表 (Course Schedule) --dfs
    private boolean isValid207 = true;
    private int[] visited207;

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        visited207 = new int[numCourses];
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] prerequisite : prerequisites) {
            graph.computeIfAbsent(prerequisite[1], k -> new ArrayList<>()).add(prerequisite[0]);
        }
        for (int i = 0; i < numCourses && isValid207; ++i) {
            if (visited207[i] == 0) {
                dfs207(i, graph);
            }
        }
        return isValid207;

    }

    private void dfs207(int u, Map<Integer, List<Integer>> graph) {
        visited207[u] = 1;
        for (int v : graph.getOrDefault(u, new ArrayList<>())) {
            if (visited207[v] == 0) {
                dfs207(v, graph);
                if (!isValid207) {
                    return;
                }
            } else if (visited207[v] == 1) {
                isValid207 = false;
                return;
            }
        }
        visited207[u] = 2;
    }

    // 207. 课程表 (Course Schedule) --拓扑排序
    public boolean canFinish2(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        int[] inDegrees = new int[numCourses];
        for (int[] prerequisite : prerequisites) {
            graph.computeIfAbsent(prerequisite[1], k -> new LinkedList<>()).add(prerequisite[0]);
            ++inDegrees[prerequisite[0]];
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; ++i) {
            if (inDegrees[i] == 0) {
                queue.offer(i);
            }
        }
        int count = 0;
        while (!queue.isEmpty()) {
            ++count;
            int cur = queue.poll();
            for (int neightbor : graph.getOrDefault(cur, new LinkedList<>())) {
                if (--inDegrees[neightbor] == 0) {
                    queue.offer(neightbor);
                }
            }
        }
        return count == numCourses;

    }

    // 210. 课程表 II (Course Schedule II) --dfs
    // 剑指 Offer II 113. 课程顺序
    private int[] visited210;
    private boolean isValid210 = true;
    private int[] res210;
    private int index210;

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        index210 = numCourses - 1;
        res210 = new int[numCourses];
        visited210 = new int[numCourses];
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] prerequisite : prerequisites) {
            graph.computeIfAbsent(prerequisite[1], k -> new ArrayList<>()).add(prerequisite[0]);
        }
        for (int i = 0; i < numCourses && isValid210; ++i) {
            if (visited210[i] == 0) {
                dfs210(i, graph);
            }

        }
        if (!isValid210) {
            return new int[0];
        }

        return res210;

    }

    private void dfs210(int u, Map<Integer, List<Integer>> graph) {
        visited210[u] = 1;
        for (int v : graph.getOrDefault(u, new ArrayList<>())) {
            if (visited210[v] == 0) {
                dfs210(v, graph);
                if (!isValid210) {
                    return;
                }
            } else if (visited210[v] == 1) {
                isValid210 = false;
                return;
            }
        }
        visited210[u] = 2;
        res210[index210--] = u;
    }

    // 210. 课程表 II (Course Schedule II) --拓扑排序
    // 剑指 Offer II 113. 课程顺序
    public int[] findOrder2(int numCourses, int[][] prerequisites) {
        int[] inDegrees = new int[numCourses];
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] prerequisite : prerequisites) {
            graph.computeIfAbsent(prerequisite[1], k -> new ArrayList<>()).add(prerequisite[0]);
            ++inDegrees[prerequisite[0]];
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < numCourses; ++i) {
            if (inDegrees[i] == 0) {
                queue.offer(i);
            }
        }
        int[] res = new int[numCourses];
        int index = 0;
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            res[index++] = cur;
            for (int neighbor : graph.getOrDefault(cur, new ArrayList<>())) {
                if (--inDegrees[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        return index == numCourses ? res : new int[0];
    }

    // 2192. 有向无环图中一个节点的所有祖先 (All Ancestors of a Node in a Directed Acyclic Graph)
    // --拓扑排序 + bfs
    public List<List<Integer>> getAncestors(int n, int[][] edges) {
        List<Set<Integer>> list = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            list.add(new HashSet<>());
        }
        List<Integer>[] g = new ArrayList[n];
        int[] deg = new int[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] e : edges) {
            g[e[0]].add(e[1]);
            ++deg[e[1]];
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (deg[i] == 0) {
                q.offer(i);
            }
        }
        while (!q.isEmpty()) {
            int x = q.poll();
            for (int y : g[x]) {
                list.get(y).add(x);
                list.get(y).addAll(list.get(x));
                if (--deg[y] == 0) {
                    q.offer(y);
                }
            }
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            Set<Integer> s = list.get(i);
            List<Integer> parents = new ArrayList<>(s);
            Collections.sort(parents);
            res.add(parents);
        }
        return res;

    }

    // 787. K 站中转内最便宜的航班 (Cheapest Flights Within K Stops) --Dijkstra
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        List<int[]>[] g = new ArrayList[n];
        for (int i = 0; i < n; ++i) {
            g[i] = new ArrayList<>();
        }
        for (int[] f : flights) {
            int a = f[0];
            int b = f[1];
            int c = f[2];
            g[a].add(new int[] { b, c });
        }
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[2] == o2[2]) {
                    return Integer.compare(o1[1], o2[1]);
                }
                return Integer.compare(o1[2], o2[2]);

            }

        });
        int res = Integer.MAX_VALUE;
        int[] cost = new int[n];
        Arrays.fill(cost, Integer.MAX_VALUE);
        // {city, cost, k}
        q.offer(new int[] { src, 0, -1 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int curCost = cur[1];
            int curK = cur[2];
            if (x == dst) {
                res = Math.min(res, curCost);
                continue;
            }
            for (int[] nei : g[x]) {
                int y = nei[0];
                int delta = nei[1];
                if (curK + 1 <= k && curCost + delta < cost[y]) {
                    cost[y] = curCost + delta;
                    q.offer(new int[] { y, cost[y], curK + 1 });
                }
            }
        }
        return res == Integer.MAX_VALUE ? -1 : res;

    }

    // 1306. 跳跃游戏 III (Jump Game III) --bfs
    public boolean canReach(int[] arr, int start) {
        int n = arr.length;
        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[n];
        queue.offer(start);
        visited[start] = true;
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            if (arr[cur] == 0) {
                return true;
            }
            int index1 = cur - arr[cur];
            if (index1 >= 0 && !visited[index1]) {
                visited[index1] = true;
                queue.offer(index1);
            }
            int index2 = cur + arr[cur];
            if (index2 < n && !visited[index2]) {
                visited[index2] = true;
                queue.offer(index2);
            }
        }
        return false;

    }

    // 1161. 最大层内元素和 (Maximum Level Sum of a Binary Tree) --bfs
    public int maxLevelSum(TreeNode root) {
        int res = 1;
        int maxSum = root.val;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int curLevel = 0;
        while (!queue.isEmpty()) {
            ++curLevel;
            int size = queue.size();
            int levelSum = 0;
            for (int i = 0; i < size; ++i) {
                TreeNode cur = queue.poll();
                levelSum += cur.val;
                if (cur.left != null) {
                    queue.offer(cur.left);
                }
                if (cur.right != null) {
                    queue.offer(cur.right);
                }
            }
            if (levelSum > maxSum) {
                maxSum = levelSum;
                res = curLevel;
            }
        }
        return res;

    }

    // 802. 找到最终的安全状态 (Find Eventual Safe States) --拓扑排序
    public List<Integer> eventualSafeNodes(int[][] graph) {
        int n = graph.length;
        Map<Integer, List<Integer>> map = new HashMap<>();
        int[] inDegrees = new int[n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < graph[i].length; ++j) {
                map.computeIfAbsent(graph[i][j], k -> new ArrayList<>()).add(i);
                ++inDegrees[i];
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (inDegrees[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            if (map.get(cur) == null) {
                continue;
            }
            for (int index : map.get(cur)) {
                if (--inDegrees[index] == 0) {
                    queue.offer(index);
                }
            }
        }
        List<Integer> res = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (inDegrees[i] == 0) {
                res.add(i);
            }
        }
        return res;
    }

    // 329. 矩阵中的最长递增路径 (Longest Increasing Path in a Matrix)
    // 剑指 Offer II 112. 最长递增路径
    private int[][] matrix329;
    private int m329;
    private int n329;
    private int[][] memo329;
    private int[][] dirs329 = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

    public int longestIncreasingPath(int[][] matrix) {
        this.m329 = matrix.length;
        this.n329 = matrix[0].length;
        this.matrix329 = matrix;
        this.memo329 = new int[m329][n329];
        for (int i = 0; i < m329; ++i) {
            Arrays.fill(memo329[i], -1);
        }
        int res = 0;
        for (int i = 0; i < m329; ++i) {
            for (int j = 0; j < n329; ++j) {
                res = Math.max(res, dfs329(i, j));
            }
        }
        return res;
    }

    private int dfs329(int i, int j) {
        if (memo329[i][j] != -1) {
            return memo329[i][j];
        }
        int max = 0;
        for (int[] d : dirs329) {
            int nx = i + d[0];
            int ny = j + d[1];
            if (nx >= 0 && nx < m329 && ny >= 0 && ny < n329 && matrix329[nx][ny] > matrix329[i][j]) {
                max = Math.max(max, dfs329(nx, ny));
            }
        }
        return memo329[i][j] = max + 1;
    }

    // 329. 矩阵中的最长递增路径 (Longest Increasing Path in a Matrix) --拓扑排序
    // 剑指 Offer II 112. 最长递增路径
    public int longestIncreasingPath2(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int[][] inDegrees = new int[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int[] direction : directions) {
                    int nx = i + direction[0];
                    int ny = j + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && matrix[i][j] < matrix[nx][ny]) {
                        ++inDegrees[nx][ny];
                    }
                }
            }
        }
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (inDegrees[i][j] == 0) {
                    queue.offer(new int[] { i, j });
                }
            }
        }
        int res = 0;
        while (!queue.isEmpty()) {
            ++res;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = queue.poll();
                for (int[] direction : directions) {
                    int nx = cur[0] + direction[0];
                    int ny = cur[1] + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && matrix[cur[0]][cur[1]] < matrix[nx][ny]) {
                        if (--inDegrees[nx][ny] == 0) {
                            queue.offer(new int[] { nx, ny });
                        }
                    }
                }
            }
        }
        return res;
    }

    // 269. 火星词典 --plus
    // 剑指 Offer II 114. 外星文字典 --拓扑排序
    public String alienOrder(String[] words) {
        int[] inDegrees = new int[26];
        int count = 0;
        Arrays.fill(inDegrees, Integer.MIN_VALUE);
        for (String word : words) {
            for (char c : word.toCharArray()) {
                if (inDegrees[c - 'a'] == Integer.MIN_VALUE) {
                    inDegrees[c - 'a'] = 0;
                    ++count;
                }
            }
        }
        Map<Character, List<Character>> map = new HashMap<>();
        for (int i = 1; i < words.length; ++i) {
            String cur = words[i];
            String pre = words[i - 1];
            int minLen = Math.min(cur.length(), pre.length());
            boolean flag = false;
            for (int j = 0; j < minLen; ++j) {
                if (pre.charAt(j) != cur.charAt(j)) {
                    flag = true;
                    if (!map.containsKey(pre.charAt(j))) {
                        map.put(pre.charAt(j), new ArrayList<>());
                    }
                    List<Character> list = map.get(pre.charAt(j));
                    if (!list.contains(cur.charAt(j))) {
                        list.add(cur.charAt(j));
                        map.put(pre.charAt(j), list);
                        ++inDegrees[cur.charAt(j) - 'a'];
                    }
                    break;
                }
            }
            if (!flag && pre.length() > cur.length()) {
                return "";
            }
        }

        Queue<Character> queue = new LinkedList<>();
        for (int i = 0; i < inDegrees.length; ++i) {
            if (inDegrees[i] == 0) {
                queue.offer((char) (i + 'a'));
            }
        }
        StringBuilder res = new StringBuilder();
        while (!queue.isEmpty()) {
            char cur = queue.poll();
            res.append(cur);
            List<Character> list = map.get(cur);
            if (list == null) {
                continue;
            }
            for (char c : list) {
                if (--inDegrees[c - 'a'] == 0) {
                    queue.offer(c);
                }
            }
        }
        return res.length() == count ? res.toString() : "";
    }

    // 2115. 从给定原材料中找到所有可以做出的菜 (Find All Possible Recipes from Given Supplies)
    // --拓扑排序
    public List<String> findAllRecipes(String[] recipes, List<List<String>> ingredients, String[] supplies) {
        Map<String, Integer> degrees = new HashMap<>();
        Map<String, List<String>> map = new HashMap<>();
        for (int i = 0; i < ingredients.size(); ++i) {
            for (String item : ingredients.get(i)) {
                map.computeIfAbsent(item, k -> new LinkedList<>()).add(recipes[i]);
            }
            degrees.put(recipes[i], ingredients.get(i).size());
        }
        List<String> res = new LinkedList<>();
        Queue<String> queue = new LinkedList<>();
        for (String supply : supplies) {
            queue.offer(supply);
        }
        while (!queue.isEmpty()) {
            String cur = queue.poll();
            if (map.get(cur) == null) {
                continue;
            }
            for (String item : map.get(cur)) {
                degrees.put(item, degrees.get(item) - 1);
                if (degrees.get(item) == 0) {
                    queue.offer(item);
                    res.add(item);
                }
            }
        }
        return res;

    }

    // 851. 喧闹和富有 (Loud and Rich) --拓扑排序
    public int[] loudAndRich(int[][] richer, int[] quiet) {
        int n = quiet.length;
        Map<Integer, List<Integer>> map = new HashMap<>();
        int[] inDegrees = new int[n];
        for (int[] rich : richer) {
            map.computeIfAbsent(rich[0], k -> new LinkedList<>()).add(rich[1]);
            ++inDegrees[rich[1]];
        }
        Queue<Integer> queue = new LinkedList<>();
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            res[i] = i;
            if (inDegrees[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            if (map.get(cur) == null) {
                continue;
            }
            for (int item : map.get(cur)) {
                if (quiet[res[cur]] < quiet[res[item]]) {
                    res[item] = res[cur];
                }
                if (--inDegrees[item] == 0) {
                    queue.offer(item);
                }
            }
        }
        return res;

    }

    // 1462. 课程表 IV (Course Schedule IV) --拓扑排序 (还需掌握floyd算法)
    public List<Boolean> checkIfPrerequisite(int numCourses, int[][] prerequisites, int[][] queries) {
        List<Integer>[] g = new ArrayList[numCourses];
        int[] d = new int[numCourses];
        for (int i = 0; i < numCourses; ++i) {
            g[i] = new ArrayList<>();
        }
        for (int[] pre : prerequisites) {
            int a = pre[0];
            int b = pre[1];
            g[a].add(b);
            ++d[b];
        }
        Map<Integer, Set<Integer>> map = new HashMap<>();
        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < numCourses; ++i) {
            if (d[i] == 0) {
                q.offer(i);
            }
            map.put(i, new HashSet<>());
        }
        while (!q.isEmpty()) {
            int x = q.poll();
            for (int y : g[x]) {
                Set<Integer> set = new HashSet<>(map.get(x));
                set.add(x);
                map.computeIfAbsent(y, k -> new HashSet<>()).addAll(set);
                if (--d[y] == 0) {
                    q.offer(y);
                }
            }
        }
        List<Boolean> res = new ArrayList<>();
        for (int[] query : queries) {
            int a = query[0];
            int b = query[1];
            res.add(map.get(b).contains(a));
        }
        return res;

    }

    // 1462. 课程表 IV (Course Schedule IV)
    private int[][] memo1462;
    private List<Integer>[] g1462;

    public List<Boolean> checkIfPrerequisite2(int numCourses, int[][] prerequisites, int[][] queries) {
        this.memo1462 = new int[numCourses][numCourses];
        this.g1462 = new ArrayList[numCourses];
        Arrays.setAll(g1462, k -> new ArrayList<>());
        for (int[] p : prerequisites) {
            g1462[p[0]].add(p[1]);
        }
        List<Boolean> res = new ArrayList<>();
        for (int[] q : queries) {
            res.add(dfs1462(q[0], q[1]));
        }
        return res;
    }

    private boolean dfs1462(int x, int target) {
        if (x == target) {
            return true;
        }
        if (memo1462[x][target] != 0) {
            return memo1462[x][target] > 0;
        }
        for (int y : g1462[x]) {
            if (dfs1462(y, target)) {
                memo1462[x][target] = 1;
                return true;
            }
        }
        memo1462[x][target] = -1;
        return false;
    }

    // 444. 重建序列 --plus
    // 剑指 Offer II 115. 重建序列 --拓扑排序
    public boolean sequenceReconstruction(int[] org, List<List<Integer>> seqs) {
        int[] inDegrees = new int[org.length + 1];
        Set<Integer> set = new HashSet<>();
        for (List<Integer> seg : seqs) {
            for (int s : seg) {
                set.add(s);
            }
        }
        if (org.length == 1 && !set.contains(1)) {
            return false;
        }
        if (set.size() != org.length) {
            return false;
        }
        Map<Integer, Set<Integer>> graph = new HashMap<>();
        for (int i = 1; i <= org.length; ++i) {
            graph.put(i, new HashSet<>());
        }
        for (List<Integer> seg : seqs) {
            for (int i = 1; i < seg.size(); ++i) {
                if (!graph.get(seg.get(i - 1)).contains(seg.get(i))) {
                    graph.get(seg.get(i - 1)).add(seg.get(i));
                    ++inDegrees[seg.get(i)];
                }

            }
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 1; i < inDegrees.length; ++i) {
            if (inDegrees[i] == 0) {
                queue.offer(i);
            }
        }
        List<Integer> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            if (queue.size() != 1) {
                return false;
            }
            int cur = queue.poll();
            res.add(cur);
            if (graph.get(cur) == null) {
                continue;
            }
            for (int item : graph.get(cur)) {
                if (--inDegrees[item] == 0) {
                    queue.offer(item);
                }
            }
        }
        if (res.size() != org.length) {
            return false;
        }
        for (int i = 0; i < res.size(); ++i) {
            if (res.get(i) != org[i]) {
                return false;
            }
        }
        return true;

    }

    // 2050. 并行课程 III (Parallel Courses III) --拓扑排序 + dp
    public int minimumTime(int n, int[][] relations, int[] time) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int[] indegrees = new int[n];
        for (int[] relation : relations) {
            int a = relation[0] - 1;
            int b = relation[1] - 1;
            map.computeIfAbsent(a, k -> new ArrayList<>()).add(b);
            ++indegrees[b];
        }

        Queue<Integer> queue = new LinkedList<>();
        int[] t = new int[n];
        for (int i = 0; i < n; ++i) {
            if (indegrees[i] == 0) {
                queue.offer(i);
            }
        }
        int res = 0;
        while (!queue.isEmpty()) {
            int x = queue.poll();
            t[x] += time[x];
            res = Math.max(res, t[x]);
            for (int y : map.getOrDefault(x, new ArrayList<>())) {
                t[y] = Math.max(t[y], t[x]);
                if (--indegrees[y] == 0) {
                    queue.offer(y);
                }
            }
        }
        return res;

    }

    // 1857. 有向图中最大颜色值 (Largest Color Value in a Directed Graph) --拓扑排序
    public int largestPathValue(String colors, int[][] edges) {
        int n = colors.length();
        int[] indegrees = new int[n];
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            map.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            ++indegrees[edge[1]];
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (indegrees[i] == 0) {
                queue.offer(i);
            }

        }
        int[][] dp = new int[n][26];
        int count = 0;
        int res = 0;
        while (!queue.isEmpty()) {
            ++count;
            int x = queue.poll();
            res = Math.max(res, ++dp[x][colors.charAt(x) - 'a']);
            for (int y : map.getOrDefault(x, new ArrayList<>())) {
                for (int i = 0; i < 26; ++i) {
                    dp[y][i] = Math.max(dp[y][i], dp[x][i]);
                }
                if (--indegrees[y] == 0) {
                    queue.offer(y);
                }
            }
        }
        if (count != n) {
            return -1;
        }
        return res;

    }

    // 785. 判断二分图 (Is Graph Bipartite?) --并查集
    // 剑指 Offer II 106. 二分图
    public boolean isBipartite2(int[][] graph) {
        int n = graph.length;
        Union785 union = new Union785(n);
        for (int i = 0; i < graph.length; ++i) {
            int[] neighbors = graph[i];
            for (int neighbor : neighbors) {
                if (union.isConnected(i, neighbor)) {
                    return false;
                }
                union.union(neighbors[0], neighbor);
            }
        }
        return true;

    }

    class Union785 {
        private int[] rank;
        private int[] parent;

        public Union785(int n) {
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

    // 785. 判断二分图 (Is Graph Bipartite?) --bfs
    // 剑指 Offer II 106. 二分图
    public boolean isBipartite(int[][] graph) {
        final int UNCOVERED = 0;
        final int GREEN = 1;
        final int RED = 2;
        int n = graph.length;
        int[] color = new int[n];
        Arrays.fill(color, UNCOVERED);
        for (int i = 0; i < n; ++i) {
            if (color[i] == UNCOVERED) {
                color[i] = RED;
                Queue<Integer> queue = new LinkedList<>();
                queue.offer(i);
                while (!queue.isEmpty()) {
                    int node = queue.poll();
                    int colorNeighborShouldBe = color[node] == RED ? GREEN : RED;
                    for (int neighbor : graph[node]) {
                        if (color[neighbor] == UNCOVERED) {
                            color[neighbor] = colorNeighborShouldBe;
                            queue.offer(neighbor);
                        } else if (color[neighbor] != colorNeighborShouldBe) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;

    }

    // 886. 可能的二分法 (Possible Bipartition) --bfs
    public boolean possibleBipartition(int n, int[][] dislikes) {
        final int UNCOVERED = 0;
        final int RED = 1;
        final int GREEN = 2;
        int[] color = new int[n];
        Arrays.fill(color, UNCOVERED);
        Map<Integer, List<Integer>> graph = buildGraph(n, dislikes);

        for (int i = 0; i < n; ++i) {
            if (color[i] == UNCOVERED) {
                color[i] = RED;
                Queue<Integer> queue = new LinkedList<>();
                queue.offer(i);
                while (!queue.isEmpty()) {
                    int node = queue.poll();
                    if (graph.get(node) == null) {
                        continue;
                    }
                    int colorNeighborShouldBe = color[node] == RED ? GREEN : RED;
                    for (int neighbor : graph.get(node)) {
                        if (color[neighbor] == UNCOVERED) {
                            color[neighbor] = colorNeighborShouldBe;
                            queue.offer(neighbor);
                        } else if (color[neighbor] != colorNeighborShouldBe) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    private Map<Integer, List<Integer>> buildGraph(int n, int[][] dislikes) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] dislike : dislikes) {
            graph.computeIfAbsent(dislike[0] - 1, k -> new LinkedList<>()).add(dislike[1] - 1);
            graph.computeIfAbsent(dislike[1] - 1, k -> new LinkedList<>()).add(dislike[0] - 1);
        }
        return graph;
    }

    // 886. 可能的二分法 (Possible Bipartition) --并查集
    public boolean possibleBipartition2(int n, int[][] dislikes) {
        Union886 union = new Union886(n);
        Map<Integer, List<Integer>> graph = buildGraph(n, dislikes);
        for (int i = 0; i < n; ++i) {
            List<Integer> neighbors = graph.get(i);
            if (neighbors == null) {
                continue;
            }
            for (int neighbor : neighbors) {
                if (union.isConnected(i, neighbor)) {
                    return false;
                }
                union.union(neighbors.get(0), neighbor);
            }
        }
        return true;

    }

    class Union886 {
        private int[] rank;
        private int[] parent;

        public Union886(int n) {
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

    // 1448. 统计二叉树中好节点的数目 (Count Good Nodes in Binary Tree) --bfs
    public int goodNodes(TreeNode root) {
        int res = 0;
        if (root == null) {
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        Queue<Integer> maxValue = new LinkedList<>();
        queue.offer(root);
        maxValue.offer(root.val);
        ++res;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                int max = maxValue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                    maxValue.offer(Math.max(node.left.val, max));
                    if (node.left.val >= max) {
                        ++res;
                    }
                }
                if (node.right != null) {
                    queue.offer(node.right);
                    maxValue.offer(Math.max(node.right.val, max));
                    if (node.right.val >= max) {
                        ++res;
                    }
                }
            }
        }
        return res;

    }

    // 1448. 统计二叉树中好节点的数目 (Count Good Nodes in Binary Tree)
    private TreeMap<Integer, Integer> map1448;
    private int res1448;

    public int goodNodes2(TreeNode root) {
        this.map1448 = new TreeMap<>();
        dfs1448(root);
        return res1448;

    }

    private void dfs1448(TreeNode root) {
        if (root == null) {
            return;
        }
        map1448.merge(root.val, 1, Integer::sum);
        if (map1448.lastKey() <= root.val) {
            ++res1448;
        }
        dfs1448(root.left);
        dfs1448(root.right);
        map1448.merge(root.val, -1, Integer::sum);
        if (map1448.get(root.val) == 0) {
            map1448.remove(root.val);
        }
    }

    // 1448. 统计二叉树中好节点的数目 (Count Good Nodes in Binary Tree)
    public int goodNodes3(TreeNode root) {
        return dfs1448(root, Integer.MIN_VALUE);

    }

    private int dfs1448(TreeNode root, int min) {
        if (root == null) {
            return 0;
        }
        int left = dfs1448(root.left, Math.max(min, root.val));
        int right = dfs1448(root.right, Math.max(min, root.val));
        return left + right + (min <= root.val ? 1 : 0);
    }

    // 1345. 跳跃游戏 IV (Jump Game IV) --bfs
    public int minJumps(int[] arr) {
        int n = arr.length;
        if (n == 1) {
            return 0;
        }
        if (arr[0] == arr[n - 1]) {
            return 1;
        }
        Map<Integer, List<Integer>> numToIndex = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            numToIndex.computeIfAbsent(arr[i], k -> new LinkedList<>()).add(i);
        }
        boolean[] visited = new boolean[n];
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { 0, 0 });
        visited[0] = true;
        while (!queue.isEmpty()) {
            int[] node = queue.poll();
            if (numToIndex.get(arr[node[0]]) != null) {
                for (int neighbor : numToIndex.get(arr[node[0]])) {
                    if (!visited[neighbor]) {
                        if (neighbor == n - 1) {
                            return node[1] + 1;
                        }
                        visited[neighbor] = true;
                        queue.offer(new int[] { neighbor, node[1] + 1 });
                    }
                }
                numToIndex.remove(arr[node[0]]);
            }

            if (node[0] - 1 >= 0 && !visited[node[0] - 1]) {
                visited[node[0] - 1] = true;
                queue.offer(new int[] { node[0] - 1, node[1] + 1 });
            }
            if (node[0] + 1 < n && !visited[node[0] + 1]) {
                if (node[0] + 1 == n - 1) {
                    return node[1] + 1;
                }
                visited[node[0] + 1] = true;
                queue.offer(new int[] { node[0] + 1, node[1] + 1 });
            }

        }
        return -1;

    }

    // 2028. 找出缺失的观测数据 (Find Missing Observations)
    public int[] missingRolls(int[] rolls, int mean, int n) {
        int length = n + rolls.length;
        int rollsSum = 0;
        for (int roll : rolls) {
            rollsSum += roll;
        }
        int missingSum = length * mean - rollsSum;
        if (n > missingSum || missingSum > 6 * n) {
            return new int[0];
        }
        int[] res = new int[n];
        int quotient = missingSum / n;
        int remainder = missingSum % n;
        for (int i = 0; i < n; ++i) {
            res[i] = quotient + (i < remainder ? 1 : 0);
        }
        return res;

    }

    // 773. 滑动谜题 (Sliding Puzzle) --bfs
    public int slidingPuzzle(int[][] board) {
        int[][] neighbors = { { 1, 3 }, { 0, 2, 4 }, { 1, 5 }, { 0, 4 }, { 1, 3, 5 }, { 2, 4 } };
        StringBuilder initalStatus = new StringBuilder();
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                initalStatus.append(board[i][j]);
            }
        }
        int res = 0;
        Queue<String> queue = new LinkedList<>();
        queue.offer(initalStatus.toString());
        Set<String> visited = new HashSet<>();
        visited.add(initalStatus.toString());
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                String node = queue.poll();
                if ("123450".equals(node)) {
                    return res;
                }
                for (String neighbor : getNeighbors773(node, neighbors)) {
                    if (!visited.contains(neighbor)) {
                        visited.add(neighbor);
                        queue.offer(neighbor);
                    }
                }
            }
            ++res;
        }
        return -1;

    }

    private List<String> getNeighbors773(String status, int[][] neighbors) {
        List<String> neightborsStatus = new ArrayList<>();

        char[] chars = status.toCharArray();
        int index = status.indexOf("0");
        for (int neightbor : neighbors[index]) {
            swap773(chars, neightbor, index);
            neightborsStatus.add(String.valueOf(chars));
            swap773(chars, neightbor, index);

        }
        return neightborsStatus;
    }

    private void swap773(char[] chars, int i, int j) {
        char temp = chars[i];
        chars[i] = chars[j];
        chars[j] = temp;
    }

    // 133. 克隆图 (Clone Graph) --bfs
    public Node cloneGraph(Node node) {
        if (node == null) {
            return node;
        }
        Node res = new Node(node.val, new ArrayList<>());
        Map<Node, Node> visited = new HashMap<>();
        Queue<Node> queue = new LinkedList<>();
        queue.offer(node);
        visited.put(node, res);
        while (!queue.isEmpty()) {
            Node cur = queue.poll();
            for (Node neighbor : cur.children) {
                if (!visited.containsKey(neighbor)) {
                    visited.put(neighbor, new Node(neighbor.val, new ArrayList<>()));
                    queue.offer(neighbor);
                }
                visited.get(cur).children.add(visited.get(neighbor));
            }
        }
        return res;

    }

    // 133. 克隆图 (Clone Graph) --dfs
    private Map<Node, Node> map133 = new HashMap<>();

    public Node cloneGraph2(Node node) {
        if (node == null) {
            return node;
        }
        if (map133.containsKey(node)) {
            return map133.get(node);
        }
        Node cloneNode = new Node(node.val, new ArrayList<>());
        map133.put(node, cloneNode);
        for (Node neighbor : node.children) {
            cloneNode.children.add(cloneGraph(neighbor));
        }
        return cloneNode;
    }

    // 1284. 转化为全零矩阵的最少反转次数 (Minimum Number of Flips to Convert Binary Matrix to
    // Zero Matrix) --bfs
    public int minFlips(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 }, { 0, 0 } };
        int inital = encode1284(mat);
        int res = 0;
        if (inital == 0) {
            return res;
        }
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(inital);
        visited.add(inital);
        while (!queue.isEmpty()) {
            ++res;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int status = queue.poll();
                int[][] curMat = decode1284(status, m, n);
                for (int x = 0; x < m; ++x) {
                    for (int y = 0; y < n; ++y) {
                        convert1284(curMat, x, y, m, n, directions);
                        int nStatus = encode1284(curMat);
                        if (nStatus == 0) {
                            return res;
                        }
                        if (!visited.contains(nStatus)) {
                            queue.offer(nStatus);
                            visited.add(nStatus);
                        }
                        convert1284(curMat, x, y, m, n, directions);
                    }
                }
            }
        }
        return -1;

    }

    private void convert1284(int[][] mat, int x, int y, int m, int n, int[][] directions) {
        for (int[] direction : directions) {
            int nx = x + direction[0];
            int ny = y + direction[1];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                mat[nx][ny] ^= 1;
            }
        }
    }

    private int[][] decode1284(int status, int m, int n) {
        int[][] res = new int[m][n];
        for (int i = m - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                res[i][j] = status & 1;
                status >>>= 1;
            }
        }
        return res;
    }

    private int encode1284(int[][] mat) {
        int res = 0;
        int m = mat.length;
        int n = mat[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                res = (res << 1) | mat[i][j];
            }
        }
        return res;
    }

    // LCP 09. 最小跳跃次数 --bfs 还需掌握 线段树、动态规划
    public int minJump(int[] jump) {
        int res = 0;
        int n = jump.length;
        int far = 0;
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(0);
        boolean[] visited = new boolean[n];
        visited[0] = true;
        while (!queue.isEmpty()) {
            ++res;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int index = queue.poll();
                for (int j = far + 1; j < index; ++j) {
                    if (!visited[j]) {
                        visited[j] = true;
                        queue.offer(j);
                    }
                }
                far = Math.max(index, far);
                if (index + jump[index] >= n) {
                    return res;
                } else if (!visited[index + jump[index]]) {
                    visited[index + jump[index]] = true;
                    queue.offer(index + jump[index]);
                }
            }
        }
        return res;

    }

    // 815. 公交路线 (Bus Routes) --bfs
    public int numBusesToDestination(int[][] routes, int source, int target) {
        if (source == target) {
            return 0;
        }

        // key : 路线
        // value : 该路线可到达的站点
        Map<Integer, Set<Integer>> graph = new HashMap<>();

        // key : 0 ~ n-1中的某一个站点
        // value : 到达该站点的步数
        Map<Integer, Integer> map = new HashMap<>();

        Queue<Integer> queue = new LinkedList<>();
        int n = routes.length;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < routes[i].length; ++j) {
                if (routes[i][j] == source) {
                    map.put(i, 1);
                    queue.offer(i);
                }
                graph.computeIfAbsent(routes[i][j], k -> new HashSet<>()).add(i);
            }
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            int step = map.get(cur);
            for (int station : routes[cur]) {
                if (station == target) {
                    return step;
                }
                if (graph.get(station) == null) {
                    continue;
                }
                for (int neighbor : graph.get(station)) {
                    if (!map.containsKey(neighbor)) {
                        map.put(neighbor, step + 1);
                        queue.offer(neighbor);
                    }
                }
            }
        }
        return -1;
    }

    // 987. 二叉树的垂序遍历 (Vertical Order Traversal of a Binary Tree) --bfs
    public List<List<Integer>> verticalTraversal(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        // key : coloum
        // val : value
        Map<Integer, List<TreeWrapNode>> map = new TreeMap<>();
        Queue<TreeWrapNode> queue = new LinkedList<>();

        queue.offer(new TreeWrapNode(0, 0, root));
        while (!queue.isEmpty()) {

            TreeWrapNode node = queue.poll();
            map.computeIfAbsent(node.colunm, k -> new ArrayList<>())
                    .add(new TreeWrapNode(node.level, node.colunm, node.node));
            if (node.node.left != null) {
                queue.offer(new TreeWrapNode(node.level + 1, node.colunm - 1, node.node.left));
            }
            if (node.node.right != null) {
                queue.offer(new TreeWrapNode(node.level + 1, node.colunm + 1, node.node.right));
            }
        }
        for (List<TreeWrapNode> list : map.values()) {
            Collections.sort(list);
            List<Integer> sub = new ArrayList<>();
            for (TreeWrapNode item : list) {
                sub.add(item.node.val);
            }
            res.add(sub);
        }
        return res;
    }

    class TreeWrapNode implements Comparable<TreeWrapNode> {
        int level;
        int colunm;
        TreeNode node;

        TreeWrapNode(int level, int colunm, TreeNode node) {
            this.level = level;
            this.colunm = colunm;
            this.node = node;
        }

        @Override
        public int compareTo(TreeWrapNode oNode) {
            return level == oNode.level ? node.val - oNode.node.val : level - oNode.level;
        }

    }

    // 987. 二叉树的垂序遍历 (Vertical Order Traversal of a Binary Tree) --dfs
    private Map<Integer, List<int[]>> map987;
    private int minCol987;

    public List<List<Integer>> verticalTraversal2(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        this.map987 = new TreeMap<>();
        dfs987(root, 0, 0);
        for (int i = minCol987; i < minCol987 + map987.size(); ++i) {
            List<int[]> list = map987.get(i);
            list.sort(new Comparator<int[]>() {

                @Override
                public int compare(int[] o1, int[] o2) {
                    if (o1[0] == o2[0]) {
                        return Integer.compare(o1[1], o2[1]);
                    }
                    return Integer.compare(o1[0], o2[0]);

                }

            });
            List<Integer> items = new ArrayList<>();
            for (int[] cur : list) {
                items.add(cur[1]);
            }
            res.add(items);
        }
        return res;
    }

    private void dfs987(TreeNode root, int i, int j) {
        if (root == null) {
            return;
        }
        minCol987 = Math.min(minCol987, j);
        map987.computeIfAbsent(j, k -> new ArrayList<>()).add(new int[] { i, root.val });
        dfs987(root.left, i + 1, j - 1);
        dfs987(root.right, i + 1, j + 1);
    }

    // 2215. 找出两数组的不同 (Find the Difference of Two Arrays)
    public List<List<Integer>> findDifference(int[] nums1, int[] nums2) {
        Set<Integer> set1 = Arrays.stream(nums1).boxed().collect(Collectors.toSet());
        Set<Integer> set2 = Arrays.stream(nums2).boxed().collect(Collectors.toSet());
        List<List<Integer>> res = new ArrayList<>();
        res.add(new ArrayList<>());
        res.add(new ArrayList<>());
        for (int num : set1) {
            if (!set2.contains(num)) {
                res.get(0).add(num);
            }
        }
        for (int num : set2) {
            if (!set1.contains(num)) {
                res.get(1).add(num);
            }
        }
        return res;
    }

    // 2216. 美化数组的最少删除数 (Minimum Deletions to Make Array Beautiful) --栈
    public int minDeletion(int[] nums) {
        int n = nums.length;
        int res = 0;
        int i = 0;
        while (i + 1 < n) {
            if (nums[i] == nums[i + 1]) {
                ++res;
                ++i;
            } else {
                i += 2;
            }
        }
        return res + (n - res) % 2;

    }

    // 2217. 找到指定长度的回文数 (Find Palindrome With Fixed Length)
    public long[] kthPalindrome(int[] queries, int intLength) {
        int n = queries.length;
        long[] res = new long[n];
        long min = (long) Math.pow(10, (intLength - 1) / 2);
        long max = (long) Math.pow(10, (intLength + 1) / 2) - 1;
        for (int i = 0; i < n; ++i) {
            if ((long) queries[i] > (long) (max - min + 1)) {
                res[i] = -1;
                continue;
            }
            int index = (int) (queries[i] - 1 + min);
            String s = String.valueOf(index);
            StringBuilder builder = new StringBuilder(s);
            if (intLength % 2 == 0) {
                s += builder.reverse().toString();
            } else {
                s += builder.reverse().toString().substring(1);
            }
            res[i] = Long.parseLong(s);
        }
        return res;
    }

    // 剑指 Offer 13. 机器人的运动范围 --bfs
    public int movingCount(int m, int n, int k) {
        int[][] directions = { { 0, 1 }, { 1, 0 } };
        int res = 0;
        boolean[][] visited = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { 0, 0 });
        visited[0][0] = true;
        ++res;
        while (!queue.isEmpty()) {
            int cur[] = queue.poll();
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]
                        && getBitsSum(nx) + getBitsSum(ny) <= k) {
                    visited[nx][ny] = true;
                    ++res;
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        return res;

    }

    private int getBitsSum(int num) {
        int res = 0;
        while (num != 0) {
            res += num % 10;
            num /= 10;
        }
        return res;
    }

    // 剑指 Offer 13. 机器人的运动范围 --递推
    public int movingCount2(int m, int n, int k) {
        if (k == 0) {
            return 1;
        }
        int res = 1;
        boolean[][] visited = new boolean[m][n];
        visited[0][0] = true;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if ((i == 0 && j == 0) || getBitsSum(i) + getBitsSum(j) > k) {
                    continue;
                }
                if (i - 1 >= 0) {
                    visited[i][j] |= visited[i - 1][j];
                }
                if (j - 1 >= 0) {
                    visited[i][j] |= visited[i][j - 1];
                }
                res += visited[i][j] ? 1 : 0;
            }
        }
        return res;

    }

    // LCP 22. 黑白方格画
    public int paintingPlan(int n, int k) {
        if (k == 0 || k == n * n) {
            return 1;
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i * n + j * n - i * j == k) {
                    res += combinationLCP22(i, n) * combinationLCP22(j, n);
                }
            }
        }
        return res;

    }

    private int combinationLCP22(int k, int n) {
        if (k == 0) {
            return 1;
        }
        int res = 1;
        for (int i = 0; i < k; ++i) {
            res *= n - i;
        }
        for (int i = 1; i <= k; ++i) {
            res /= i;
        }
        return res;
    }

    // LCP 45. 自行车炫技赛场 --bfs
    public int[][] bicycleYard(int[] position, int[][] terrain, int[][] obstacle) {
        int m = terrain.length;
        int n = terrain[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { position[0], position[1], 1 });
        boolean[][][] visited = new boolean[m][n][102];
        visited[position[0]][position[1]][1] = true;
        while (!queue.isEmpty()) {
            int[] p = queue.poll();
            for (int[] direction : directions) {
                int nx = p[0] + direction[0];
                int ny = p[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int h1 = terrain[p[0]][p[1]];
                    int h2 = terrain[nx][ny];
                    int o2 = obstacle[nx][ny];
                    int nSpeedChanged = h1 - h2 - o2;
                    int nSpeed = p[2] + nSpeedChanged;
                    if (nSpeed > 0 && !visited[nx][ny][nSpeed]) {
                        visited[nx][ny][nSpeed] = true;
                        queue.offer(new int[] { nx, ny, nSpeed });
                    }

                }
            }
        }
        List<int[]> res = new ArrayList<>();
        visited[position[0]][position[1]][1] = false;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (visited[i][j][1]) {
                    res.add(new int[] { i, j });
                }
            }
        }
        return res.toArray(new int[res.size()][]);
    }

    // 452. 用最少数量的箭引爆气球 (Minimum Number of Arrows to Burst Balloons) --排序 + 贪心
    public int findMinArrowShots(int[][] points) {
        Arrays.sort(points, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.valueOf(o1[0]).compareTo(Integer.valueOf(o2[0]));
            }
        });
        int res = 1;
        int min = points[0][1];
        for (int i = 1; i < points.length; ++i) {
            if (points[i][0] > min) {
                ++res;
                min = points[i][1];
            } else {
                min = Math.min(min, points[i][1]);
            }
        }
        return res;
    }

    // 757. 设置交集大小至少为2 (Set Intersection Size At Least Two)
    public int intersectionSizeTwo(int[][] intervals) {
        int n = intervals.length;
        Arrays.sort(intervals, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return o2[1] - o1[1];
                }
                return o1[0] - o2[0];
            }

        });
        int res = 2;
        int cur = intervals[n - 1][0];
        int next = cur + 1;
        for (int i = n - 2; i >= 0; --i) {
            if (intervals[i][1] >= next) {
                continue;
            }
            if (intervals[i][1] >= cur && intervals[i][1] < next) {
                res += 1;
                next = cur;
                cur = intervals[i][0];
            } else if (intervals[i][1] < cur) {
                res += 2;
                cur = intervals[i][0];
                next = cur + 1;
            }

        }
        return res;

    }

    // 1376. 通知所有员工所需的时间 (Time Needed to Inform All Employees)
    public int numOfMinutes(int n, int headID, int[] manager, int[] informTime) {
        Map<Integer, List<Integer>> g = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            if (manager[i] != -1) {
                g.computeIfAbsent(manager[i], k -> new ArrayList<>()).add(i);
            }
        }
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] { headID, informTime[headID] });
        int res = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = q.poll();
                int x = cur[0];
                int t = cur[1];
                res = Math.max(res, t);
                for (int y : g.getOrDefault(x, new ArrayList<>())) {
                    q.offer(new int[] { y, t + informTime[y] });
                    res = Math.max(res, t + informTime[y]);
                }
            }
        }
        return res;

    }

    // 1376. 通知所有员工所需的时间 (Time Needed to Inform All Employees)
    private Map<Integer, List<Integer>> g1376;

    private int[] informTime1376;
    // private int res;

    public int numOfMinutes2(int n, int headID, int[] manager, int[] informTime) {
        this.g1376 = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            if (manager[i] != -1) {
                g1376.computeIfAbsent(manager[i], k -> new ArrayList<>()).add(i);
            }
        }
        this.informTime1376 = informTime;
        // 自底向上
        return dfs1376(headID);
        // 自顶向下
        // dfs1376(headID,informTime[headID]);
        // return res;

    }

    // 自顶向下
    // private void dfs1376(int x, int t) {
    // res = Math.max(res, t);
    // for (int y : g.getOrDefault(x, new ArrayList<>())) {
    // dfs1376(y, t + informTime[y]);
    // }
    // }

    // 自底向上
    private int dfs1376(int x) {
        int max = 0;
        for (int y : g1376.getOrDefault(x, new ArrayList<>())) {
            max = Math.max(max, dfs1376(y));
        }
        return max + informTime1376[x];
    }

    // 301. 删除无效的括号 (Remove Invalid Parentheses) --bfs
    public List<String> removeInvalidParentheses(String s) {
        Queue<String> queue = new LinkedList<>();
        queue.offer(s);
        List<String> res = new LinkedList<>();
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                String cur = queue.poll();
                if (isValid301(cur)) {
                    res.add(cur);
                }
                queue.offer(cur);
            }
            if (!res.isEmpty()) {
                return res;
            }
            Set<String> set = new HashSet<>();
            for (int i = 0; i < size; ++i) {
                String cur = queue.poll();
                for (int j = 0; j < cur.length(); ++j) {
                    if (cur.charAt(j) == '(' || cur.charAt(j) == ')') {
                        String nString = cur.substring(0, j) + cur.substring(j + 1);
                        if (!set.contains(nString)) {
                            set.add(nString);
                            queue.offer(nString);
                        }
                    }
                }
            }
        }
        res.add("");
        return res;

    }

    private boolean isValid301(String string) {
        int count = 0;
        for (char c : string.toCharArray()) {
            if (c == '(') {
                ++count;
            } else if (c == ')') {
                --count;
                if (count < 0) {
                    return false;
                }
            }
        }

        return count == 0;
    }

    // 728. 自除数 (Self Dividing Numbers)
    public List<Integer> selfDividingNumbers(int left, int right) {
        List<Integer> res = new ArrayList<>();
        for (int i = left; i <= right; ++i) {
            if (isValid728(i)) {
                res.add(i);
            }
        }
        return res;

    }

    private boolean isValid728(int num) {
        int numCopy = num;
        while (numCopy != 0) {
            int bit = numCopy % 10;
            if (bit == 0 || num % bit != 0) {
                return false;
            }
            numCopy /= 10;
        }
        return true;
    }

    // 397. 整数替换 (Integer Replacement) --bfs 还需掌握 贪心、dfs
    public int integerReplacement(int n) {
        Queue<long[]> queue = new LinkedList<>();
        queue.offer(new long[] { (long) n, 0L });
        Set<Long> visited = new HashSet<>();
        visited.add((long) n);
        while (!queue.isEmpty()) {
            long cur[] = queue.poll();
            if (cur[0] == 1L) {
                return (int) cur[1];
            }
            if (cur[0] % 2 == 0L) {
                if (visited.add((long) (cur[0] / 2))) {
                    queue.offer(new long[] { (long) (cur[0] / 2), cur[1] + 1 });
                }
            } else {
                if (visited.add((long) (cur[0] - 1))) {
                    queue.offer(new long[] { (long) (cur[0] - 1), cur[1] + 1 });
                }
                if (visited.add((long) (cur[0] + 1))) {
                    queue.offer(new long[] { (long) (cur[0] + 1), cur[1] + 1 });
                }
            }
        }
        return -1;

    }

    // 1657. 确定两个字符串是否接近 (Determine if Two Strings Are Close)
    public boolean closeStrings(String word1, String word2) {
        if (word1.length() != word2.length()) {
            return false;
        }
        int[] counts1 = new int[26];
        int[] counts2 = new int[26];
        for (int i = 0; i < word1.length(); ++i) {
            ++counts1[word1.charAt(i) - 'a'];
            ++counts2[word2.charAt(i) - 'a'];
        }
        for (int i = 0; i < 26; ++i) {
            // 一个为0 一个不为0
            if ((counts1[i] == 0) ^ (counts2[i] == 0)) {
                return false;
            }
        }
        Arrays.sort(counts1);
        Arrays.sort(counts2);
        return Arrays.equals(counts1, counts2);

    }

    // 647. 回文子串 (Palindromic Substrings) --dp 还需掌握 中心扩展法
    // 剑指 Offer II 020. 回文子字符串的个数
    public int countSubstrings(String s) {
        int res = 0;
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                if (s.charAt(i) == s.charAt(j) && (j - i < 2 || dp[i + 1][j - 1])) {
                    ++res;
                    dp[i][j] = true;
                }
            }
        }
        return res;

    }

    // 400. 第 N 位数字 (Nth Digit) --模拟 还需掌握 二分查找
    // 剑指 Offer 44. 数字序列中某一位的数字
    public int findNthDigit(int n) {
        int bits = 1;
        int count = 9;
        while (n > (long) bits * count) {
            n -= bits * count;
            ++bits;
            count *= 10;
        }
        --n;
        int num = (int) Math.pow(10, bits - 1) + n / bits;
        int numIndex = n % bits;
        return (num / (int) Math.pow(10, bits - numIndex - 1)) % 10;

    }

    // 6033. 转换数字的最少位翻转次数 (Minimum Bit Flips to Convert Number)
    public int minBitFlips(int start, int goal) {
        int diff = start ^ goal;
        return Integer.bitCount(diff);
    }

    // 2221. 数组的三角和 (Find Triangular Sum of an Array)
    public int triangularSum(int[] nums) {
        int n = nums.length;
        for (int i = 1; i < n; ++i) {
            for (int j = 0; j < n - i; ++j) {
                nums[j] = (nums[j] + nums[j + 1]) % 10;
            }
        }
        return nums[0];
    }

    // 2222. 选择建筑的方案数 (Number of Ways to Select Buildings)
    public long numberOfWays(String s) {
        int n = s.length();
        long res = 0L;
        int[] right0 = new int[n];
        for (int i = n - 2; i >= 0; --i) {
            right0[i] = right0[i + 1] + (s.charAt(i + 1) == '0' ? 1 : 0);
        }
        long left0 = 0L;
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) == '1') {
                res += left0 * right0[i];
            } else {
                res += (i - left0) * (n - i - 1 - right0[i]);
                ++left0;
            }
        }
        return res;

    }

    // 6055. 转化时间需要的最少操作数 (Minimum Number of Operations to Convert Time)
    public int convertTime(String current, String correct) {
        String[] time1 = current.split(":");
        String[] time2 = correct.split(":");
        int res = 0;
        int m2 = Integer.parseInt(time2[1]) + Integer.parseInt(time2[0]) * 60;
        int m1 = Integer.parseInt(time1[1]) + Integer.parseInt(time1[0]) * 60;
        int diff = m2 - m1;
        int[] divides = { 60, 15, 5, 1 };
        for (int divide : divides) {
            res += diff / divide;
            diff %= divide;
        }
        return res;

    }

    // 2225. 找出输掉零场或一场比赛的玩家 (Find Players With Zero or One Losses)
    public List<List<Integer>> findWinners(int[][] matches) {
        List<List<Integer>> res = new ArrayList<>();
        Map<Integer, Integer> lose = new HashMap<>();
        for (int[] match : matches) {
            lose.merge(match[1], 1, Integer::sum);
        }
        Set<Integer> res0 = new TreeSet<>();
        Set<Integer> res1 = new TreeSet<>();
        for (int[] match : matches) {
            if (!lose.containsKey(match[0])) {
                res0.add(match[0]);
            }
            if (lose.getOrDefault(match[1], 0) == 1) {
                res1.add(match[1]);
            }
        }
        List<Integer> ret0 = new ArrayList<>(res0);
        Collections.sort(ret0);
        List<Integer> ret1 = new ArrayList<>(res1);
        Collections.sort(ret1);
        res.add(ret0);
        res.add(ret1);
        return res;

    }

    // 5219. 每个小孩最多能分到多少糖果 (Maximum Candies Allocated to K Children)
    public int maximumCandies(int[] candies, long k) {
        long max = Long.MIN_VALUE;
        for (int candy : candies) {
            max = Math.max(candy, max);
        }
        long min = 1;
        long res = 0;
        while (min <= max) {
            long mid = min + ((max - min) >>> 1);
            if (canDivided(candies, k, mid)) {
                res = mid;
                min = mid + 1;
            } else {
                max = mid - 1;
            }
        }
        return (int) (res);

    }

    private boolean canDivided(int[] candies, long k, long unit) {
        long res = 0;
        for (int candy : candies) {
            res += candy / unit;
            if (res >= k) {
                return true;
            }
        }
        return false;
    }

    // 1042. 不邻接植花 (Flower Planting With No Adjacent)
    public int[] gardenNoAdj(int n, int[][] paths) {
        int[] res = new int[n];
        List<Integer>[] g = new ArrayList[n];
        for (int i = 0; i < n; ++i) {
            g[i] = new ArrayList<>();
        }
        for (int[] p : paths) {
            int a = p[0] - 1;
            int b = p[1] - 1;
            g[a].add(b);
            g[b].add(a);
        }
        for (int i = 0; i < n; ++i) {
            if (res[i] != 0) {
                continue;
            }
            int mask = (1 << 5) - 2;
            for (int y : g[i]) {
                if ((mask & (1 << res[y])) != 0) {
                    mask ^= 1 << res[y];
                }
            }
            res[i] = Integer.numberOfTrailingZeros(mask);
        }
        return res;

    }

    // 529. 扫雷游戏 (Minesweeper) --bfs
    public char[][] updateBoard(char[][] board, int[] click) {
        int x = click[0];
        int y = click[1];
        if (board[x][y] == 'M') {
            board[x][y] = 'X';
            return board;
        }
        int m = board.length;
        int n = board[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 }, { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 } };
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { x, y });
        boolean[][] visited = new boolean[m][n];
        while (!queue.isEmpty()) {
            int[] p = queue.poll();
            int count = 0;
            for (int[] direction : directions) {
                int nx = p[0] + direction[0];
                int ny = p[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && board[nx][ny] == 'M') {
                    ++count;
                }
            }
            if (count > 0) {
                board[p[0]][p[1]] = (char) (count + '0');
            } else {
                board[p[0]][p[1]] = 'B';
                for (int[] direction : directions) {
                    int nx = p[0] + direction[0];
                    int ny = p[1] + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]) {
                        visited[nx][ny] = true;
                        queue.offer(new int[] { nx, ny });
                    }
                }
            }
        }
        return board;

    }

    // 1466. 重新规划路线 (Reorder Routes to Make All Paths Lead to the City Zero)
    private List<int[]>[] g1466;
    private int res1466;

    public int minReorder1466(int n, int[][] connections) {
        this.g1466 = new ArrayList[n];
        Arrays.setAll(g1466, k -> new ArrayList<>());
        for (int[] c : connections) {
            g1466[c[0]].add(new int[] { c[1], 1 });
            g1466[c[1]].add(new int[] { c[0], 0 });
        }
        dfs1466(0, -1);
        return res1466;

    }

    private void dfs1466(int x, int fa) {
        for (int[] y : g1466[x]) {
            if (y[0] != fa) {
                res1466 += y[1];
                dfs1466(y[0], x);
            }
        }
    }

    // 623. 在二叉树中增加一行 (Add One Row to Tree) --bfs
    public TreeNode addOneRow(TreeNode root, int val, int depth) {
        if (depth == 1) {
            TreeNode res = new TreeNode(val, root, null);
            return res;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int level = 0;
        while (!queue.isEmpty()) {
            ++level;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (level == depth - 1) {
                    TreeNode tempLeft = node.left;
                    node.left = new TreeNode(val);
                    node.left.left = tempLeft;

                    TreeNode tempRight = node.right;
                    node.right = new TreeNode(val);
                    node.right.right = tempRight;
                } else {
                    if (node.left != null) {
                        queue.offer(node.left);
                    }
                    if (node.right != null) {
                        queue.offer(node.right);
                    }
                }
            }
            if (level == depth - 1) {
                break;
            }
        }
        return root;

    }

    // 863. 二叉树中所有距离为 K 的结点 (All Nodes Distance K in Binary Tree) --bfs
    public List<Integer> distanceK(TreeNode root, TreeNode target, int k) {
        List<Integer> res = new LinkedList<>();
        if (k == 0) {
            res.add(target.val);
            return res;
        }
        Map<Integer, List<Integer>> graph = new HashMap<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (!graph.containsKey(node.val)) {
                    graph.put(node.val, new LinkedList<>());
                }
                if (node.left != null) {
                    graph.computeIfAbsent(node.val, o -> new LinkedList<>()).add(node.left.val);
                    graph.computeIfAbsent(node.left.val, o -> new LinkedList<>()).add(node.val);
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    graph.computeIfAbsent(node.val, o -> new LinkedList<>()).add(node.right.val);
                    graph.computeIfAbsent(node.right.val, o -> new LinkedList<>()).add(node.val);
                    queue.offer(node.right);
                }
            }
        }

        int level = 0;
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue2 = new LinkedList<>();
        queue2.offer(target.val);
        visited.add(target.val);
        while (!queue2.isEmpty()) {
            ++level;
            int size = queue2.size();
            for (int i = 0; i < size; ++i) {
                int cur = queue2.poll();
                if (graph.get(cur) == null) {
                    continue;
                }
                for (int neighbor : graph.get(cur)) {
                    if (!visited.contains(neighbor)) {
                        visited.add(neighbor);
                        queue2.offer(neighbor);
                    }
                }
            }
            if (level == k) {
                res.addAll(queue2);
                break;
            }
        }
        return res;

    }

    // 958. 二叉树的完全性检验 (Check Completeness of a Binary Tree) --bfs
    public boolean isCompleteTree(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        boolean flag = false;
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.left != null) {
                if (flag) {
                    return false;
                }
                queue.offer(node.left);
            } else {
                flag = true;
            }

            if (node.right != null) {
                if (flag) {
                    return false;
                }
                queue.offer(node.right);
            } else {
                flag = true;
            }
        }
        return true;
    }

    // 655. 输出二叉树 (Print Binary Tree) --bfs
    public List<List<String>> printTree(TreeNode root) {
        int m = getHeight(root);
        int n = (1 << m) - 1;
        List<List<String>> res = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            List<String> list = new ArrayList<>();
            for (int j = 0; j < n; ++j) {
                list.add("");
            }
            res.add(list);
        }
        Queue<Node655> queue = new LinkedList<>();
        queue.offer(new Node655(root, 0, n >> 1));
        while (!queue.isEmpty()) {
            Node655 node = queue.poll();
            res.get(node.r).set(node.c, node.node.val + "");
            if (node.node.left != null) {
                queue.offer(new Node655(node.node.left, node.r + 1, node.c - (1 << (m - 1 - node.r - 1))));
            }
            if (node.node.right != null) {
                queue.offer(new Node655(node.node.right, node.r + 1, node.c + (1 << (m - 1 - node.r - 1))));
            }
        }
        return res;

    }

    public class Node655 {
        TreeNode node;
        int r;
        int c;

        public Node655(TreeNode node, int r, int c) {
            this.node = node;
            this.r = r;
            this.c = c;
        }
    }

    private int getHeight(TreeNode root) {
        int height = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            ++height;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }

                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
        }

        return height;
    }

    // 662. 二叉树最大宽度 (Maximum Width of Binary Tree) --bfs
    public int widthOfBinaryTree(TreeNode root) {
        int res = 1;
        Queue<TreeNode> queue = new LinkedList<>();
        root.val = 1;
        queue.offer(root);
        while (!queue.isEmpty()) {
            int min = -1;
            int max = -1;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    node.left.val = node.val << 1;
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    node.right.val = (node.val << 1) + 1;
                    queue.offer(node.right);
                }
                if (i == 0) {
                    min = node.val;
                }
                if (i == size - 1) {
                    max = node.val;
                }
            }
            res = Math.max(res, max - min + 1);
        }
        return res;

    }

    // 1293. 网格中的最短路径 (Shortest Path in a Grid with Obstacles Elimination) --bfs
    public int shortestPath(int[][] grid, int k) {
        int m = grid.length;
        int n = grid[0].length;
        if (m == 1 && n == 1) {
            return 0;
        }
        if (k >= m + n - 3) {
            return m + n - 2;
        }
        k = Math.min(m + n - 3, k);
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { 0, 0, k });
        int[][] visited = new int[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                visited[i][j] = -1;
            }
        }
        visited[0][0] = k;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int res = 0;
        while (!queue.isEmpty()) {
            ++res;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = queue.poll();
                for (int[] direction : directions) {
                    int nx = cur[0] + direction[0];
                    int ny = cur[1] + direction[1];
                    int remain = cur[2];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                        if (nx == m - 1 && ny == n - 1) {
                            return res;
                        }
                        remain = grid[nx][ny] == 0 ? remain : --remain;
                        if (remain >= 0) {
                            if (visited[nx][ny] == -1 || remain > visited[nx][ny]) {
                                visited[nx][ny] = remain;
                                queue.offer(new int[] { nx, ny, remain });
                            }
                        }
                    }
                }
            }
        }
        return -1;
    }

    // 1298. 你能从盒子里获得的最大糖果数 (Maximum Candies You Can Get from Boxes) --bfs
    public int maxCandies(int[] status, int[] candies, int[][] keys, int[][] containedBoxes, int[] initialBoxes) {
        int n = status.length;
        int res = 0;
        // 是否使用过
        boolean[] used_boxes = new boolean[n];
        // 是否拥有箱子📦
        boolean[] have_boxes = new boolean[n];
        // 是否拥有钥匙🔑
        boolean[] can_open = new boolean[n];
        for (int i = 0; i < n; ++i) {
            can_open[i] = status[i] == 1;
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int initialBox : initialBoxes) {
            have_boxes[initialBox] = true;
            if (can_open[initialBox]) {
                res += candies[initialBox];
                used_boxes[initialBox] = true;
                queue.offer(initialBox);
            }
        }
        while (!queue.isEmpty()) {
            int bigBox = queue.poll();
            for (int key : keys[bigBox]) {
                can_open[key] = true;
                if (!used_boxes[key] && have_boxes[key]) {
                    used_boxes[key] = true;
                    queue.offer(key);
                    res += candies[key];
                }
            }
            for (int box : containedBoxes[bigBox]) {
                have_boxes[box] = true;
                if (!used_boxes[box] && can_open[box]) {
                    used_boxes[box] = true;
                    queue.offer(box);
                    res += candies[box];
                }
            }
        }
        return res;

    }

    // 剑指 Offer II 043. 往完全二叉树添加节点
    // 919. 完全二叉树插入器 (Complete Binary Tree Inserter) --bfs
    class CBTInserter {
        private Queue<TreeNode> queue;
        private TreeNode root;

        public CBTInserter(TreeNode root) {
            this.root = root;
            this.queue = new LinkedList<>();
            queue.offer(root);
            while (!queue.isEmpty()) {
                TreeNode node = queue.peek();
                if (node.left == null) {
                    break;
                }
                if (node.right == null) {
                    queue.offer(node.left);
                    break;
                }
                node = queue.poll();
                queue.offer(node.left);
                queue.offer(node.right);
            }
        }

        public int insert(int val) {
            TreeNode node = queue.peek();
            if (node.left == null) {
                node.left = new TreeNode(val);
                queue.offer(node.left);
            } else {
                node.right = new TreeNode(val);
                queue.offer(node.right);
                node = queue.poll();
            }
            return node.val;
        }

        public TreeNode get_root() {
            return root;
        }
    }

    // 1261. 在受污染的二叉树中查找元素 (Find Elements in a Contaminated Binary Tree)
    class FindElements {
        private Set<Integer> set;

        public FindElements(TreeNode root) {
            this.set = new HashSet<>();
            dfs(root, 0);
        }

        private void dfs(TreeNode root, int x) {
            if (root == null) {
                return;
            }
            set.add(x);
            dfs(root.left, x * 2 + 1);
            dfs(root.right, x * 2 + 2);
        }

        public boolean find(int target) {
            return set.contains(target);
        }
    }

    // 1261. 在受污染的二叉树中查找元素 (Find Elements in a Contaminated Binary Tree) --不用set
    class FindElements2 {
        private TreeNode root;

        public FindElements2(TreeNode root) {
            this.root = root;
        }

        public boolean find(int target) {
            ++target;
            TreeNode node = root;
            for (int i = 30 - Integer.numberOfLeadingZeros(target); i >= 0; --i) {
                int bit = (target >> i) & 1;
                node = bit == 0 ? node.left : node.right;
                if (node == null) {
                    return false;
                }
            }
            return true;
        }
    }

    // 6037. 按奇偶性交换后的最大数字 (Largest Number After Digit Swaps by Parity)
    public int largestInteger(int num) {
        char[] chars = String.valueOf(num).toCharArray();
        int n = chars.length;
        PriorityQueue<Integer> oddQueue = new PriorityQueue<>((o1, o2) -> o2 - o1);
        PriorityQueue<Integer> evenQueue = new PriorityQueue<>((o1, o2) -> o2 - o1);

        for (int i = 0; i < chars.length; ++i) {
            int cur = chars[i] - '0';
            if (cur % 2 == 0) {
                evenQueue.offer(cur);
            } else {
                oddQueue.offer(cur);
            }
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            int cur = chars[i] - '0';
            if ((cur & 1) == 0) {
                res = res * 10 + evenQueue.poll();
            } else {
                res = res * 10 + oddQueue.poll();
            }
        }
        return res;
    }

    // 2232. 向表达式添加括号后的最小结果 (Minimize Result by Adding Parentheses to Expression)
    public String minimizeResult(String expression) {
        int pos = expression.indexOf("+");
        int minVal = Integer.MAX_VALUE;
        int n = expression.length();
        String res = "";
        for (int i = 0; i < pos; ++i) {
            for (int j = pos + 1; j < n; ++j) {
                String s1 = expression.substring(0, i);
                String s2 = expression.substring(i, j + 1);
                String s3 = expression.substring(j + 1);
                int cur = getRes2232(s1, s2, s3);
                if (cur < minVal) {
                    minVal = cur;
                    res = String.format("%s(%s)%s", s1, s2, s3);
                }
            }
        }
        return res;

    }

    private int getRes2232(String s1, String s2, String s3) {
        String[] split = s2.split("\\+");
        int res = Integer.parseInt(split[0]) + Integer.parseInt(split[1]);
        if (!s1.isEmpty()) {
            res *= Integer.parseInt(s1);
        }
        if (!s3.isEmpty()) {
            res *= Integer.parseInt(s3);
        }
        return res;
    }

    // 6039. K 次增加后的最大乘积 (Maximum Product After K Increments)
    public int maximumProduct(int[] nums, int k) {
        final int MOD = 1000000007;
        PriorityQueue<Integer> priorityQueue = new PriorityQueue<>();
        for (int num : nums) {
            priorityQueue.offer(num);
        }
        while (k-- > 0) {
            int cur = priorityQueue.poll();
            priorityQueue.offer(cur + 1);
        }
        long res = 1L;
        while (!priorityQueue.isEmpty()) {
            res = ((res % MOD) * (priorityQueue.poll() % MOD)) % MOD;
        }
        return (int) res;
    }

    // 297. 二叉树的序列化与反序列化 (Serialize and Deserialize Binary Tree)
    // 剑指 Offer 37. 序列化二叉树
    // 剑指 Offer II 048. 序列化与反序列化二叉树
    // 449. 序列化和反序列化二叉搜索树 (Serialize and Deserialize BST) --bfs
    public class Codec {

        // Encodes a tree to a single string.
        public String serialize(TreeNode root) {
            if (root == null) {
                return "#";
            }
            StringBuilder builder = new StringBuilder();
            Queue<TreeNode> queue = new LinkedList<>();
            queue.offer(root);
            while (!queue.isEmpty()) {
                TreeNode node = queue.poll();
                if (node == null) {
                    builder.append("#").append(",");
                    continue;
                }
                builder.append(node.val).append(",");
                queue.offer(node.left);
                queue.offer(node.right);
            }
            return builder.toString();
        }

        // Decodes your encoded data to tree.
        public TreeNode deserialize(String data) {
            if ("#".equals(data)) {
                return null;
            }
            String[] strings = data.split(",");
            Queue<TreeNode> queue = new LinkedList<>();
            TreeNode root = new TreeNode(Integer.parseInt(strings[0]));
            queue.offer(root);
            int index = 1;
            while (index < strings.length) {
                TreeNode node = queue.poll();
                if (!"#".equals(strings[index])) {
                    node.left = new TreeNode(Integer.parseInt(strings[index]));
                    queue.offer(node.left);
                }
                ++index;
                if (!"#".equals(strings[index])) {
                    node.right = new TreeNode(Integer.parseInt(strings[index]));
                    queue.offer(node.right);
                }
                ++index;
            }
            return root;
        }
    }

    // 617. 合并二叉树 (Merge Two Binary Trees) --bfs
    public TreeNode mergeTrees(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return root2;
        }
        if (root2 == null) {
            return root1;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        TreeNode root = new TreeNode(root1.val + root2.val);
        queue.offer(root);
        Queue<TreeNode> queue1 = new LinkedList<>();
        queue1.offer(root1);
        Queue<TreeNode> queue2 = new LinkedList<>();
        queue2.offer(root2);
        while (!queue1.isEmpty() && !queue2.isEmpty()) {
            TreeNode node = queue.poll();
            TreeNode node1 = queue1.poll();
            TreeNode node2 = queue2.poll();
            if (node1.left != null || node2.left != null) {
                if (node1.left != null && node2.left != null) {
                    node.left = new TreeNode(node1.left.val + node2.left.val);
                    queue.offer(node.left);
                    queue1.offer(node1.left);
                    queue2.offer(node2.left);
                } else if (node1.left != null) {
                    node.left = node1.left;
                } else if (node2.left != null) {
                    node.left = node2.left;
                }
            }

            if (node1.right != null || node2.right != null) {
                if (node1.right != null && node2.right != null) {
                    node.right = new TreeNode(node1.right.val + node2.right.val);
                    queue.offer(node.right);
                    queue1.offer(node1.right);
                    queue2.offer(node2.right);
                } else if (node1.right != null) {
                    node.right = node1.right;
                } else if (node2.right != null) {
                    node.right = node2.right;
                }
            }
        }
        return root;

    }

    // 617. 合并二叉树 (Merge Two Binary Trees)
    public TreeNode mergeTrees2(TreeNode root1, TreeNode root2) {
        return dfs617(root1, root2);
    }

    private TreeNode dfs617(TreeNode root1, TreeNode root2) {
        if (root1 == null) {
            return root2;
        }
        if (root2 == null) {
            return root1;
        }
        TreeNode node = new TreeNode(root1.val + root2.val);
        node.left = dfs617(root1.left, root2.left);
        node.right = dfs617(root1.right, root2.right);
        return node;
    }

    // 1609. 奇偶树 (Even Odd Tree) --bfs
    public boolean isEvenOddTree(TreeNode root) {
        int level = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            int pre = level % 2 == 0 ? Integer.MIN_VALUE : Integer.MAX_VALUE;
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                // 奇数层的值应为偶数
                // 偶数层的值应为奇数
                if (((node.val & 1) ^ (level & 1)) == 0) {
                    return false;
                }
                // 偶数层应严格递增
                // 奇数层应严格递减
                if ((level & 1) == 0 && node.val <= pre || (level & 1) == 1 && node.val >= pre) {
                    return false;
                }
                pre = node.val;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            ++level;
        }
        return true;

    }

    // 993. 二叉树的堂兄弟节点 (Cousins in Binary Tree) --bfs
    public boolean isCousins(TreeNode root, int x, int y) {
        if (root.val == x || root.val == y) {
            return false;
        }
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            int fa = -1;
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    if (node.left.val == x || node.left.val == y) {
                        if (fa != -1) {
                            return true;
                        }
                        fa = node.val;
                    }
                    queue.offer(node.left);
                }

                if (node.right != null) {
                    if (node.right.val == x || node.right.val == y) {
                        if (fa != -1) {
                            return fa != node.val;
                        }
                        fa = node.val;
                    }
                    queue.offer(node.right);
                }
            }
            if (fa != -1) {
                return false;
            }
        }
        return false;
    }

    // 993. 二叉树的堂兄弟节点 (Cousins in Binary Tree) --dfs
    private int x_fa993;
    private int x_level993;
    private int x993;
    private int y993;
    private boolean res993;

    public boolean isCousins2(TreeNode root, int x, int y) {
        this.x_fa993 = -2;
        this.x_level993 = -2;
        this.x993 = x;
        this.y993 = y;
        dfs993(root, 0, -1);
        return res993;

    }

    private void dfs993(TreeNode node, int level, int fa) {
        if (node == null) {
            return;
        }
        if (node.val == x993 || node.val == y993) {
            if (x_fa993 != -2) {
                res993 = level == x_level993 && x_fa993 != fa;
                return;
            }
            x_fa993 = fa;
            x_level993 = level;
        }
        dfs993(node.left, level + 1, node.val);
        if (res993) {
            return;
        }
        dfs993(node.right, level + 1, node.val);
    }

    // 222. 完全二叉树的节点个数 (Count Complete Tree Nodes)
    public int countNodes(TreeNode root) {
        int left = 0;
        int right = 50000;
        int res = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            TreeNode node = root;
            if (hasNode222(node, mid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private boolean hasNode222(TreeNode node, int target) {
        int bit = Integer.highestOneBit(target) >> 1;
        while (bit != 0 && node != null) {
            if ((target & bit) == 0) {
                node = node.left;
            } else {
                node = node.right;
            }
            bit >>= 1;
        }
        return node != null;
    }

    // 284. 顶端迭代器 (Peeking Iterator)
    class PeekingIterator implements Iterator<Integer> {
        private Iterator<Integer> iterator;
        private Integer nextInteger;

        public PeekingIterator(Iterator<Integer> iterator) {
            // initialize any member here.
            this.iterator = iterator;
            this.nextInteger = iterator.next();

        }

        // Returns the next element in the iteration without advancing the iterator.
        public Integer peek() {
            return nextInteger;

        }

        // hasNext() and next() should behave the same as in the Iterator interface.
        // Override them if needed.
        @Override
        public Integer next() {
            Integer res = nextInteger;
            nextInteger = iterator.hasNext() ? iterator.next() : null;
            return res;

        }

        @Override
        public boolean hasNext() {
            return nextInteger != null;
        }
    }

    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        backtrack113(res, path, targetSum, root);
        return res;

    }

    // 113. 路径总和 II (Path Sum II) --回溯
    // 剑指 Offer 34. 二叉树中和为某一值的路径
    private void backtrack113(List<List<Integer>> res, List<Integer> path, int targetSum, TreeNode root) {
        if (root == null) {
            return;
        }
        path.add(root.val);
        targetSum -= root.val;
        if (root.left == null && root.right == null && targetSum == 0) {
            res.add(new ArrayList<>(path));
        }
        backtrack113(res, path, targetSum, root.left);
        backtrack113(res, path, targetSum, root.right);
        path.remove(path.size() - 1);
    }

    // 113. 路径总和 II (Path Sum II) --bfs
    // 剑指 Offer 34. 二叉树中和为某一值的路径
    private List<List<Integer>> res113 = new ArrayList<>();

    public List<List<Integer>> pathSum2(TreeNode root, int targetSum) {
        if (root == null) {
            return res113;
        }
        Map<TreeNode, TreeNode> map = new HashMap<>();
        Queue<TreeNode> queueNode = new LinkedList<>();
        Queue<Integer> queueSum = new LinkedList<>();
        queueNode.offer(root);
        queueSum.offer(0);
        while (!queueNode.isEmpty()) {
            TreeNode node = queueNode.poll();
            int sum = node.val + queueSum.poll();
            if (node.left == null && node.right == null && sum == targetSum) {
                res113.add(getPath113(node, map));
            }
            if (node.left != null) {
                queueNode.offer(node.left);
                queueSum.offer(sum);
                map.put(node.left, node);
            }
            if (node.right != null) {
                queueNode.offer(node.right);
                queueSum.offer(sum);
                map.put(node.right, node);
            }
        }
        return res113;
    }

    private List<Integer> getPath113(TreeNode node, Map<TreeNode, TreeNode> map) {
        List<Integer> res = new ArrayList<>();
        while (node != null) {
            res.add(node.val);
            node = map.get(node);
        }
        Collections.reverse(res);
        return res;
    }

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

    // 236. 二叉树的最近公共祖先 (Lowest Common Ancestor of a Binary Tree) --bfs
    // 剑指 Offer 68 - II. 二叉树的最近公共祖先
    // 面试题 04.08. 首个共同祖先 (First Common Ancestor LCCI) --bfs
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        Map<TreeNode, TreeNode> map = new HashMap<>();
        int count = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (p == node || q == node) {
                if (++count == 2) {
                    break;
                }
            }
            if (node.left != null) {
                queue.offer(node.left);
                map.put(node.left, node);
            }
            if (node.right != null) {
                queue.offer(node.right);
                map.put(node.right, node);
            }
        }
        return getAncestor236(p, q, map);

    }

    private TreeNode getAncestor236(TreeNode p, TreeNode q, Map<TreeNode, TreeNode> map) {
        TreeNode pCopy = p;
        TreeNode qCopy = q;
        while (qCopy != pCopy) {
            pCopy = pCopy == null ? q : map.get(pCopy);
            qCopy = qCopy == null ? p : map.get(qCopy);
        }
        return qCopy;
    }

    // 236. 二叉树的最近公共祖先 (Lowest Common Ancestor of a Binary Tree) --dfs
    // 剑指 Offer 68 - II. 二叉树的最近公共祖先
    // 面试题 04.08. 首个共同祖先 (First Common Ancestor LCCI) --dfs
    public TreeNode lowestCommonAncestor2(TreeNode root, TreeNode p, TreeNode q) {
        if (root == p || root == q || root == null) {
            return root;
        }
        TreeNode left = lowestCommonAncestor2(root.left, p, q);
        TreeNode right = lowestCommonAncestor2(root.right, p, q);
        if (left == null) {
            return right;
        }
        if (right == null) {
            return left;
        }
        return root;

    }

    // 1311. 获取你好友已观看的视频 (Get Watched Videos by Your Friends) --bfs
    public List<String> watchedVideosByFriends(List<List<String>> watchedVideos, int[][] friends, int id, int level) {
        int n = watchedVideos.size();
        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[n];
        queue.offer(id);
        visited[id] = true;
        while (!queue.isEmpty()) {
            --level;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int cur = queue.poll();
                for (int neightbor : friends[cur]) {
                    if (!visited[neightbor]) {
                        visited[neightbor] = true;
                        queue.offer(neightbor);
                    }
                }
            }
            if (level == 0) {
                break;
            }
        }
        Map<String, Integer> map = new HashMap<>();
        while (!queue.isEmpty()) {
            int index = queue.poll();
            for (String video : watchedVideos.get(index)) {
                map.put(video, map.getOrDefault(video, 0) + 1);
            }
        }
        List<String> res = new ArrayList<>();
        for (String video : map.keySet()) {
            res.add(video);
        }

        res = res.stream().sorted(new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                return map.get(o1) != map.get(o2) ? map.get(o1) - map.get(o2) : o1.compareTo(o2);
            }
        }).collect(Collectors.toList());
        return res;
    }

    // 310. 最小高度树 (Minimum Height Trees) --拓扑排序
    public List<Integer> findMinHeightTrees(int n, int[][] edges) {
        if (n == 1) {
            return List.of(0);
        }
        int[] deg = new int[n];
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] edge : edges) {
            ++deg[edge[0]];
            ++deg[edge[1]];
            g[edge[0]].add(edge[1]);
            g[edge[1]].add(edge[0]);
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            deg[i] -= 1;
            if (deg[i] == 0) {
                queue.offer(i);
            }
        }
        List<Integer> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            int size = queue.size();
            res.clear();
            for (int i = 0; i < size; ++i) {
                int x = queue.poll();
                res.add(x);
                for (int y : g[x]) {
                    if (--deg[y] == 0) {
                        queue.offer(y);
                    }
                }
            }
        }
        return res;

    }

    // 2039. 网络空闲的时刻 (The Time When the Network Becomes Idle) --bfs
    public int networkBecomesIdle(int[][] edges, int[] patience) {
        int n = patience.length;
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new LinkedList<>()).add(edge[1]);
            graph.computeIfAbsent(edge[1], k -> new LinkedList<>()).add(edge[0]);
        }
        boolean[] visited = new boolean[n];
        Queue<Integer> queue = new LinkedList<>();
        visited[0] = true;
        queue.offer(0);
        int level = 1;
        int res = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int cur = queue.poll();
                if (graph.get(cur) == null) {
                    continue;
                }
                for (int neighbor : graph.get(cur)) {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        queue.offer(neighbor);
                        int val = patience[neighbor] * ((2 * level - 1) / patience[neighbor]) + 2 * level + 1;
                        res = Math.max(res, val);
                    }
                }
            }
            ++level;
        }
        return res;

    }

    // 380. O(1) 时间插入、删除和获取随机元素 (Insert Delete GetRandom O(1))
    // 剑指 Offer II 030. 插入、删除和随机访问都是 O(1) 的容器
    class RandomizedSet {
        private List<Integer> list;
        private Map<Integer, Integer> map;
        private Random random;

        /** Initialize your data structure here. */
        public RandomizedSet() {
            list = new ArrayList<>();
            map = new HashMap<>();
            random = new Random();
        }

        /**
         * Inserts a value to the set. Returns true if the set did not already contain
         * the specified element.
         */
        public boolean insert(int val) {
            if (map.containsKey(val)) {
                return false;
            }
            int index = list.size();
            map.put(val, index);
            list.add(val);
            return true;
        }

        /**
         * Removes a value from the set. Returns true if the set contained the specified
         * element.
         */
        public boolean remove(int val) {
            if (!map.containsKey(val)) {
                return false;
            }
            int index = map.get(val);
            int last = list.get(list.size() - 1);

            list.set(index, last);
            list.remove(list.size() - 1);

            map.put(last, index);
            map.remove(val);
            return true;
        }

        /** Get a random element from the set. */
        public int getRandom() {
            int index = random.nextInt(list.size());
            return list.get(index);
        }
    }

    // 399. 除法求值 --并查集
    // 剑指 Offer II 111. 计算除法
    public double[] calcEquation(List<List<String>> equations, double[] values, List<List<String>> queries) {
        Map<String, Integer> map = new HashMap<>();
        Union399 union = new Union399(equations.size() * 2);
        int index = 0;
        int i = 0;
        for (List<String> equation : equations) {
            String s1 = equation.get(0);
            String s2 = equation.get(1);
            if (!map.containsKey(s1)) {
                map.put(s1, index++);
            }
            if (!map.containsKey(s2)) {
                map.put(s2, index++);
            }
            union.union(map.get(s1), map.get(s2), values[i++]);
        }
        double[] res = new double[queries.size()];
        i = 0;
        for (List<String> query : queries) {
            String s1 = query.get(0);
            String s2 = query.get(1);
            Integer index1 = map.get(s1);
            Integer index2 = map.get(s2);
            if (index1 == null || index2 == null) {
                res[i++] = -1.0d;
            } else {
                res[i++] = union.isConnected(index1, index2);
            }
        }
        return res;

    }

    public class Union399 {
        private double[] weight;
        private int[] parent;

        public Union399(int n) {
            weight = new double[n];
            Arrays.fill(weight, 1.0d);
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
        }

        public int getRoot(int p) {
            if (parent[p] != p) {
                int origin = parent[p];
                parent[p] = getRoot(parent[p]);
                weight[p] *= weight[origin];
            }
            return parent[p];
        }

        public double isConnected(int p1, int p2) {
            int root1 = getRoot(p1);
            int root2 = getRoot(p2);
            if (root1 == root2) {
                return weight[p1] / weight[p2];
            }
            return -1.0d;
        }

        public void union(int p1, int p2, double value) {
            int root1 = getRoot(p1);
            int root2 = getRoot(p2);
            parent[root1] = root2;
            weight[root1] = weight[p2] * value / weight[p1];
        }
    }

    // 399. 除法求值 --bfs
    // 剑指 Offer II 111. 计算除法
    public double[] calcEquation2(List<List<String>> equations, double[] values, List<List<String>> queries) {
        int count = 0;
        // 映射
        Map<String, Integer> map = new HashMap<>();
        for (List<String> equation : equations) {
            if (!map.containsKey(equation.get(0))) {
                map.put(equation.get(0), count++);
            }

            if (!map.containsKey(equation.get(1))) {
                map.put(equation.get(1), count++);
            }
        }
        // 邻接表
        int n = values.length;
        Map<Integer, List<Bean399>> edges = new HashMap<>();

        for (int i = 0; i < n; ++i) {
            int a = map.get(equations.get(i).get(0));
            int b = map.get(equations.get(i).get(1));
            edges.computeIfAbsent(a, k -> new LinkedList<>()).add(new Bean399(b, values[i]));
            edges.computeIfAbsent(b, k -> new LinkedList<>()).add(new Bean399(a, 1.0d / values[i]));
        }
        double[] res = new double[queries.size()];
        for (int i = 0; i < res.length; ++i) {
            double ans = -1d;
            if (map.containsKey(queries.get(i).get(0)) && map.containsKey(queries.get(i).get(1))) {
                int a = map.get(queries.get(i).get(0));
                int b = map.get(queries.get(i).get(1));
                if (a == b) {
                    ans = 1d;
                } else {
                    Queue<Integer> queue = new LinkedList<>();
                    double[] ratios = new double[count];
                    queue.offer(a);
                    Arrays.fill(ratios, -1d);
                    ratios[a] = 1d;
                    while (!queue.isEmpty() && ratios[b] < 0) {
                        int x = queue.poll();
                        if (edges.get(x) == null) {
                            continue;
                        }
                        for (Bean399 bean399 : edges.get(x)) {
                            int y = bean399.index;
                            if (ratios[y] < 0) {
                                ratios[y] = ratios[x] * bean399.val;
                                queue.offer(y);
                            }
                        }
                    }
                    ans = ratios[b];
                }
            }
            res[i] = ans;
        }
        return res;
    }

    class Bean399 {
        int index;
        double val;

        Bean399(int index, double val) {
            this.index = index;
            this.val = val;
        }
    }

    // 1129. 颜色交替的最短路径 (Shortest Path with Alternating Colors)
    public int[] shortestAlternatingPaths(int n, int[][] redEdges, int[][] blueEdges) {
        List<Integer>[][] g = new ArrayList[2][n];
        int[][] dis = new int[2][n];
        final int BLUE = 0;
        final int RED = 1;
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            g[BLUE][i] = new ArrayList<>();
            g[RED][i] = new ArrayList<>();
            dis[BLUE][i] = Integer.MAX_VALUE;
            dis[RED][i] = Integer.MAX_VALUE;
        }
        for (int[] e : redEdges) {
            int a = e[0];
            int b = e[1];
            g[RED][a].add(b);
        }
        for (int[] e : blueEdges) {
            int a = e[0];
            int b = e[1];
            g[BLUE][a].add(b);
        }

        dis[BLUE][0] = 0;
        dis[RED][0] = 0;
        Queue<int[]> q = new PriorityQueue<>((o1, o2) -> Integer.compare(o1[2], o2[2]));
        q.offer(new int[] { 0, BLUE, 0 });
        q.offer(new int[] { 0, RED, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int color = cur[1];
            int d = cur[2];
            if (d > dis[color][x]) {
                continue;
            }
            for (int y : g[color ^ 1][x]) {
                if (d + 1 < dis[color ^ 1][y]) {
                    dis[color ^ 1][y] = d + 1;
                    q.offer(new int[] { y, color ^ 1, d + 1 });
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            int min = Math.min(dis[RED][i], dis[BLUE][i]);
            res[i] = min == Integer.MAX_VALUE ? -1 : min;
        }
        return res;

    }

    // 2059. 转化数字的最小运算数 (Minimum Operations to Convert Number) --bfs
    public int minimumOperations(int[] nums, int start, int goal) {
        boolean[] visited = new boolean[1001];
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(start);
        visited[start] = true;
        int level = 0;
        while (!queue.isEmpty()) {
            ++level;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int x = queue.poll();
                for (int num : nums) {
                    int added = x + num;
                    int sub = x - num;
                    int xor = x ^ num;
                    if (added == goal || sub == goal || xor == goal) {
                        return level;
                    }
                    if (added <= 1000 && added >= 0) {
                        if (!visited[added]) {
                            visited[added] = true;
                            queue.offer(added);
                        }
                    }
                    if (sub <= 1000 && sub >= 0) {
                        if (!visited[sub]) {
                            visited[sub] = true;
                            queue.offer(sub);
                        }
                    }
                    if (xor <= 1000 && xor >= 0) {
                        if (!visited[xor]) {
                            visited[xor] = true;
                            queue.offer(xor);
                        }
                    }
                }
            }
        }
        return -1;

    }

    // 2146. 价格范围内最高排名的 K 样物品 (K Highest Ranked Items Within a Price Range) --bfs
    public List<List<Integer>> highestRankedKItems(int[][] grid, int[] pricing, int[] start, int k) {
        int[][] directions = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        List<int[]> list = new ArrayList<>();
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { start[0], start[1], 0 });
        if (grid[start[0]][start[1]] >= pricing[0] && grid[start[0]][start[1]] <= pricing[1]) {
            list.add(new int[] { start[0], start[1], 0, grid[start[0]][start[1]] });
        }
        grid[start[0]][start[1]] = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = queue.poll();
                int x = cur[0];
                int y = cur[1];
                int step = cur[2];
                for (int[] direction : directions) {
                    int nx = x + direction[0];
                    int ny = y + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] != 0) {
                        if (grid[nx][ny] >= pricing[0] && grid[nx][ny] <= pricing[1]) {
                            list.add(new int[] { nx, ny, step + 1, grid[nx][ny] });
                        }
                        grid[nx][ny] = 0;
                        queue.offer(new int[] { nx, ny, step + 1 });
                    }
                }
            }
            if (list.size() >= k) {
                break;
            }
        }
        Collections.sort(list, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[2] != o2[2]) {
                    return Integer.compare(o1[2], o2[2]);
                }
                if (o1[3] != o2[3]) {
                    return Integer.compare(o1[3], o2[3]);
                }
                if (o1[0] != o2[0]) {
                    return Integer.compare(o1[0], o2[0]);
                }
                return Integer.compare(o1[1], o2[1]);
            }

        });

        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < Math.min(k, list.size()); ++i) {
            res.add(List.of(list.get(i)[0], list.get(i)[1]));
        }
        return res;

    }

    // 2101. 引爆最多的炸弹 (Detonate the Maximum Bombs) --bfs
    private List<Integer>[] g2101;
    private int n2101;
    private int[][] bombs2101;

    public int maximumDetonation(int[][] bombs) {
        this.n2101 = bombs.length;
        this.g2101 = new ArrayList[n2101];
        this.bombs2101 = bombs;
        Arrays.setAll(g2101, k -> new ArrayList<>());
        for (int i = 0; i < n2101; ++i) {
            for (int j = 0; j < n2101; ++j) {
                if (i == j) {
                    continue;
                }
                if (checkDistance2101(i, j)) {
                    g2101[i].add(j);
                }
            }
        }
        int res = 0;
        for (int i = 0; i < n2101; ++i) {
            res = Math.max(res, cal2101(i));
        }
        return res;
    }

    private int cal2101(int start) {
        int res = 0;
        boolean[] vis = new boolean[n2101];
        vis[start] = true;
        Queue<Integer> q = new LinkedList<>();
        q.offer(start);
        while (!q.isEmpty()) {
            ++res;
            int x = q.poll();
            for (int y : g2101[x]) {
                if (!vis[y]) {
                    vis[y] = true;
                    q.offer(y);
                }
            }
        }
        return res;
    }

    private boolean checkDistance2101(int x, int y) {
        return (long) (bombs2101[y][1] - bombs2101[x][1]) * (bombs2101[y][1] - bombs2101[x][1])
                + (long) (bombs2101[y][0] - bombs2101[x][0])
                        * (bombs2101[y][0] - bombs2101[x][0]) <= (long) bombs2101[x][2] * bombs2101[x][2];
    }

    // 2239. 找到最接近 0 的数字 (Find Closest Number to Zero)
    public int findClosestNumber(int[] nums) {
        int res = -100000;
        for (int num : nums) {
            if (Math.abs(num) < Math.abs(res)) {
                res = num;
            } else if (Math.abs(num) == Math.abs(res) && num > res) {
                res = num;
            }
        }
        return res;

    }

    // 2239. 找到最接近 0 的数字 (Find Closest Number to Zero)
    public int findClosestNumber2(int[] nums) {
        Integer[] arr = Arrays.stream(nums).boxed().toArray(Integer[]::new);
        Arrays.sort(arr, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                if (Math.abs(o1) == Math.abs(o2)) {
                    return Integer.compare(o2, o1);
                }
                return Integer.compare(Math.abs(o1), Math.abs(o2));
            }

        });
        return arr[0];

    }

    // 6061. 买钢笔和铅笔的方案数 (Number of Ways to Buy Pens and Pencils)
    public long waysToBuyPensPencils(int total, int cost1, int cost2) {
        long res = 0L;
        long penCounts = 0L;
        while (penCounts * cost1 <= total) {
            long remain = total - penCounts * cost1;
            res += remain / cost2 + 1;
            ++penCounts;
        }
        return res;

    }

    // 2241. 设计一个 ATM 机器 (Design an ATM Machine)
    class ATM {
        private int[] remains;
        private int[] map = new int[] { 20, 50, 100, 200, 500 };

        public ATM() {
            this.remains = new int[5];

        }

        public void deposit(int[] banknotesCount) {
            for (int i = 0; i < 5; ++i) {
                remains[i] += banknotesCount[i];
            }
        }

        public int[] withdraw(int amount) {
            int[] res = new int[5];
            for (int i = 4; i >= 0; --i) {
                int m = map[i];
                int cnt = Math.min(remains[i], amount / m);
                res[i] += cnt;
                amount -= cnt * m;
            }
            if (amount > 0) {
                return new int[] { -1 };
            }
            for (int i = 4; i >= 0; --i) {
                remains[i] -= res[i];
            }
            return res;
        }
    }

    // 6070. 计算字符串的数字和 (Calculate Digit Sum of a String)
    public String digitSum(String s, int k) {
        if (s.length() <= k) {
            return s;
        }
        while (s.length() > k) {
            s = getRes6070(s, k);
        }
        return s;

    }

    private String getRes6070(String s, int k) {
        StringBuilder builder = new StringBuilder();
        StringBuilder cur = new StringBuilder();
        for (int i = 0; i < s.length(); ++i) {
            cur.append(s.charAt(i));
            if (cur.length() == k) {
                int num = getNum6070(cur.toString());
                builder.append(num);
                cur.setLength(0);
            }
        }
        if (cur.length() != 0) {
            int num = getNum6070(cur.toString());
            builder.append(num);
            cur.setLength(0);
        }

        return builder.toString();
    }

    private int getNum6070(String s) {
        int res = 0;
        for (char c : s.toCharArray()) {
            res += c - '0';
        }
        return res;
    }

    // 2244. 完成所有任务需要的最少轮数 (Minimum Rounds to Complete All Tasks)
    public int minimumRounds(int[] tasks) {
        int res = 0;
        Map<Integer, Integer> cnts = new HashMap<>();
        for (int t : tasks) {
            cnts.merge(t, 1, Integer::sum);
        }
        for (int v : cnts.values()) {
            if (v == 1) {
                return -1;
            }
            res += (v + 2) / 3;
        }
        return res;

    }

    // 675. 为高尔夫比赛砍树 (Cut Off Trees for Golf Event) --bfs
    public int cutOffTree(List<List<Integer>> forest) {
        int m = forest.size();
        int n = forest.get(0).size();

        List<int[]> trees = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int val = forest.get(i).get(j);
                if (val > 1) {
                    trees.add(new int[] { i, j, val });
                }
            }
        }
        Collections.sort(trees, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }

        });
        int sr = 0;
        int sc = 0;
        int res = 0;
        for (int[] tree : trees) {
            int distance = getDistance675(forest, sr, sc, tree[0], tree[1]);
            if (distance == -1) {
                return -1;
            }
            res += distance;
            sr = tree[0];
            sc = tree[1];
        }
        return res;

    }

    private int getDistance675(List<List<Integer>> forest, int sr, int sc, int tr, int tc) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = forest.size();
        int n = forest.get(0).size();
        Queue<int[]> queue = new LinkedList<>();
        boolean[][] visited = new boolean[m][n];
        queue.offer(new int[] { sr, sc, 0 });
        visited[sr][sc] = true;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            if (cur[0] == tr && cur[1] == tc) {
                return cur[2];
            }
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny] && forest.get(nx).get(ny) > 0) {
                    visited[nx][ny] = true;
                    queue.offer(new int[] { nx, ny, cur[2] + 1 });
                }
            }
        }
        return -1;
    }

    // 面试题 05.03. Reverse Bits LCCI --滑动窗口
    public int reverseBits(int num) {
        int right = 0;
        int res = 0;
        int count = 0;
        for (int left = 0; left < 32; ++left) {
            if ((num & (1 << left)) == 0) {
                ++count;
                while (count > 1) {
                    if ((num & (1 << right)) == 0) {
                        --count;
                    }
                    ++right;
                }
            }
            res = Math.max(res, left - right + 1);
        }
        return res;

    }

    // 1971. 寻找图中是否存在路径 (Find if Path Exists in Graph) --bfs
    public boolean validPath(int n, int[][] edges, int source, int destination) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new LinkedList<>()).add(edge[1]);
            graph.computeIfAbsent(edge[1], k -> new LinkedList<>()).add(edge[0]);
        }
        boolean[] visited = new boolean[n];
        visited[source] = true;
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(source);
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            if (cur == destination) {
                return true;
            }
            if (graph.get(cur) == null) {
                continue;
            }
            for (int neighbor : graph.get(cur)) {
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    queue.offer(neighbor);
                }
            }
        }
        return false;

    }

    // 1971. 寻找图中是否存在路径 (Find if Path Exists in Graph) --并查集
    public boolean validPath2(int n, int[][] edges, int source, int destination) {
        if (source == destination) {
            return true;
        }
        Union1971 union = new Union1971(n);
        for (int[] edge : edges) {
            union.union(edge[0], edge[1]);
            if (union.isConnected(source, destination)) {
                return true;
            }
        }
        return false;

    }

    class Union1971 {
        private int[] rank;
        private int[] parent;

        Union1971(int n) {
            rank = new int[n];
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
                rank[i] = 1;
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
        }
    }

    // 1457. 二叉树中的伪回文路径 (Pseudo-Palindromic Paths in a Binary Tree) --bfs
    public int pseudoPalindromicPaths(TreeNode root) {
        root.val = 1 << root.val;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        int res = 0;
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.left == null && node.right == null) {
                if (Integer.bitCount(node.val) <= 1) {
                    ++res;
                }
            }
            if (node.left != null) {
                node.left.val = node.val ^ (1 << node.left.val);
                queue.offer(node.left);
            }
            if (node.right != null) {
                node.right.val = node.val ^ (1 << node.right.val);
                queue.offer(node.right);
            }
        }
        return res;

    }

    // 1457. 二叉树中的伪回文路径 (Pseudo-Palindromic Paths in a Binary Tree) --dfs
    private int res1457;

    public int pseudoPalindromicPaths2(TreeNode root) {
        dfs1457(root, 0);
        return res1457;
    }

    private void dfs1457(TreeNode root, int d) {
        if (root == null) {
            return;
        }
        d ^= 1 << root.val;
        if (root.left == null && root.right == null) {
            res1457 += Integer.bitCount(d) <= 1 ? 1 : 0;
            return;
        }
        dfs1457(root.left, d);
        dfs1457(root.right, d);
    }

    // 865. 具有所有最深节点的最小子树 (Smallest Subtree with all the Deepest Nodes) --bfs
    // 1123. 最深叶节点的最近公共祖先 (Lowest Common Ancestor of Deepest Leaves)
    public TreeNode lcaDeepestLeaves(TreeNode root) {
        Map<TreeNode, TreeNode> map = new HashMap<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        Queue<TreeNode> queue2 = new LinkedList<>();
        while (!queue.isEmpty()) {
            int size = queue.size();
            queue2.clear();
            queue2.addAll(queue);
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                    map.put(node.left, node);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                    map.put(node.right, node);
                }
            }
        }
        Set<TreeNode> set = new HashSet<>();
        while (queue2.size() != 1) {
            TreeNode node = queue2.poll();
            TreeNode parent = map.get(node);
            if (!set.contains(parent)) {
                set.add(parent);
                queue2.offer(parent);
            }
        }

        return queue2.poll();

    }

    // 865. 具有所有最深节点的最小子树 (Smallest Subtree with all the Deepest Nodes)
    // 1123. 最深叶节点的最近公共祖先 (Lowest Common Ancestor of Deepest Leaves)
    private int maxDepth865;
    private TreeNode res865;

    public TreeNode lcaDeepestLeaves2(TreeNode root) {
        maxDepth865 = -1;
        dfs865(root, 0);
        return res865;

    }

    private int dfs865(TreeNode root, int d) {
        if (root == null) {
            maxDepth865 = Math.max(maxDepth865, d);
            return d;
        }
        int left = dfs865(root.left, d + 1);
        int right = dfs865(root.right, d + 1);
        if (left == right && left == maxDepth865) {
            res865 = root;
        }
        return Math.max(left, right);

    }

    // 1091. 二进制矩阵中的最短路径 (Shortest Path in Binary Matrix) --bfs
    public int shortestPathBinaryMatrix(int[][] grid) {
        int n = grid.length;
        if (grid[0][0] == 1 || grid[n - 1][n - 1] == 1) {
            return -1;
        }
        grid[0][0] = 1;
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] { 0, 0, 1 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            int c = cur[2];
            if (x == n - 1 && y == n - 1) {
                return c;
            }
            for (int i = -1; i <= 1; ++i) {
                for (int j = -1; j <= 1; ++j) {
                    int nx = x + i;
                    int ny = y + j;
                    if (nx >= 0 && nx < n && ny >= 0 && ny < n && grid[nx][ny] == 0) {
                        grid[nx][ny] = 1;
                        q.offer(new int[] { nx, ny, c + 1 });
                    }
                }
            }
        }
        return -1;

    }

    // 1210. 穿过迷宫的最少移动次数 (Minimum Moves to Reach Target with Rotations) --bfs
    public int minimumMoves(int[][] grid) {
        int n = grid.length;
        if (grid[n - 1][n - 1] == 1 || grid[n - 1][n - 2] == 1) {
            return -1;
        }
        final int HORIZONTAL = 0;
        final int VERTICAL = 1;
        boolean[][][] vis = new boolean[n][n][2];
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] { 0, 0, HORIZONTAL, 0 });
        vis[0][0][HORIZONTAL] = true;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int y = cur[1];
            int orientation = cur[2];
            int step = cur[3];
            if (x == n - 1 && y == n - 2 && orientation == HORIZONTAL) {
                return step;
            }
            if (orientation == HORIZONTAL) {
                if (y + 2 < n && grid[x][y + 2] == 0 && !vis[x][y + 1][HORIZONTAL]) {
                    vis[x][y + 1][HORIZONTAL] = true;
                    q.offer(new int[] { x, y + 1, HORIZONTAL, step + 1 });
                }
                if (x + 1 < n && y + 1 < n && grid[x + 1][y] == 0 && grid[x + 1][y + 1] == 0
                        && !vis[x + 1][y][HORIZONTAL]) {
                    vis[x + 1][y][HORIZONTAL] = true;
                    q.offer(new int[] { x + 1, y, HORIZONTAL, step + 1 });
                }
                if (x + 1 < n && y + 1 < n && grid[x + 1][y] == 0 && grid[x + 1][y + 1] == 0 && !vis[x][y][VERTICAL]) {
                    vis[x][y][VERTICAL] = true;
                    q.offer(new int[] { x, y, VERTICAL, step + 1 });
                }
            } else {
                if (x + 2 < n && grid[x + 2][y] == 0 && !vis[x + 1][y][VERTICAL]) {
                    vis[x + 1][y][VERTICAL] = true;
                    q.offer(new int[] { x + 1, y, VERTICAL, step + 1 });
                }
                if (x + 1 < n && y + 1 < n && grid[x][y + 1] == 0 && grid[x + 1][y + 1] == 0
                        && !vis[x][y + 1][VERTICAL]) {
                    vis[x][y + 1][VERTICAL] = true;
                    q.offer(new int[] { x, y + 1, VERTICAL, step + 1 });
                }
                if (x + 1 < n && y + 1 < n && grid[x][y + 1] == 0 && grid[x + 1][y + 1] == 0
                        && !vis[x][y][HORIZONTAL]) {
                    vis[x][y][HORIZONTAL] = true;
                    q.offer(new int[] { x, y, HORIZONTAL, step + 1 });
                }
            }
        }
        return -1;

    }

    // 1368. 使网格图至少有一条有效路径的最小代价 (Minimum Cost to Make at Least One Valid Path in a
    // Grid) --SPFA
    public int minCost(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        // 是否在队列中
        boolean[][] inTheQueue = new boolean[m][n];
        inTheQueue[0][0] = true;
        int[][] distance = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(distance[i], 0x3f3f3f3f);
        }
        distance[0][0] = 0;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { 0, 0 });
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            inTheQueue[x][y] = false;
            for (int i = 0; i < directions.length; ++i) {
                int nx = x + directions[i][0];
                int ny = y + directions[i][1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    int newDistance = distance[x][y] + ((grid[x][y] == i + 1) ? 0 : 1);
                    if (newDistance < distance[nx][ny]) {
                        distance[nx][ny] = newDistance;
                        if (!inTheQueue[nx][ny]) {
                            inTheQueue[nx][ny] = true;
                            queue.offer(new int[] { nx, ny });
                        }
                    }
                }
            }
        }
        return distance[m - 1][n - 1];

    }

    // 1368. 使网格图至少有一条有效路径的最小代价 (Minimum Cost to Make at Least One Valid Path in a
    // Grid) --Dijkstra
    public int minCost2(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

        int[][] distance = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(distance[i], Integer.MAX_VALUE >> 1);
        }
        distance[0][0] = 0;

        // index0 : x ; index1 : y ; index3 : distance
        PriorityQueue<int[]> priorityQueue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }

        });
        priorityQueue.offer(new int[] { 0, 0, 0 });
        while (!priorityQueue.isEmpty()) {
            int[] cur = priorityQueue.poll();
            int x = cur[0];
            int y = cur[1];
            int dist = cur[2];
            if (visited[x][y]) {
                continue;
            }
            visited[x][y] = true;
            for (int i = 0; i < directions.length; ++i) {
                int nx = x + directions[i][0];
                int ny = y + directions[i][1];
                int nDist = dist + (grid[x][y] == i + 1 ? 0 : 1);
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && nDist < distance[nx][ny]) {
                    distance[nx][ny] = nDist;
                    priorityQueue.offer(new int[] { nx, ny, nDist });
                }
            }
        }
        return distance[m - 1][n - 1];
    }

    // 1368. 使网格图至少有一条有效路径的最小代价 (Minimum Cost to Make at Least One Valid Path in a
    // Grid) -- 0-1 bfs
    public int minCost3(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
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
            int originalDirection = grid[x][y] - 1;
            if (x == m - 1 && y == n - 1) {
                return d;
            }
            for (int i = 0; i < 4; ++i) {
                int nx = directions[i][0] + x;
                int ny = directions[i][1] + y;
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    if (i == originalDirection) {
                        if (d < dis[nx][ny]) {
                            dis[nx][ny] = d;
                            deque.offerFirst(new int[] { nx, ny });
                        }
                    } else {
                        if (d + 1 < dis[nx][ny]) {
                            dis[nx][ny] = d + 1;
                            deque.offerLast(new int[] { nx, ny });
                        }
                    }
                }
            }
        }
        return dis[m - 1][n - 1];
    }

    // 864. 获取所有钥匙的最短路径 (Shortest Path to Get All Keys) --bfs
    public int shortestPathAllKeys(String[] grid) {
        int m = grid.length;
        int n = grid[0].length();
        int keyMask = 0;
        int startX = 0;
        int startY = 0;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                char c = grid[i].charAt(j);
                if (c == '@') {
                    startX = i;
                    startY = j;
                } else if (Character.isLowerCase(c)) {
                    keyMask |= 1 << (c - 'a');
                }
            }
        }
        Queue<int[]> queue = new LinkedList<>();
        // x, y, key, step;
        queue.offer(new int[] { startX, startY, 0, 0 });
        boolean[][][] visited = new boolean[m][n][1 << 6];
        visited[startX][startY][0] = true;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            int curMask = cur[2];
            int step = cur[3];
            if (curMask == keyMask) {
                return step;
            }
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    char c = grid[nx].charAt(ny);
                    // 遇到了墙
                    if (c == '#') {
                        continue;
                    }
                    // 遇到了锁🔒 但没有钥匙🔑
                    if (Character.isUpperCase(c) && (curMask & (1 << (c - 'A'))) == 0) {
                        continue;
                    }
                    if (Character.isLowerCase(c)) {
                        int nMask = curMask | (1 << (c - 'a'));
                        if (!visited[nx][ny][nMask]) {
                            visited[nx][ny][nMask] = true;
                            queue.offer(new int[] { nx, ny, nMask, step + 1 });
                        }
                    } else {
                        if (!visited[nx][ny][curMask]) {
                            visited[nx][ny][curMask] = true;
                            queue.offer(new int[] { nx, ny, curMask, step + 1 });

                        }

                    }
                }
            }
        }
        return -1;
    }

    // 396. 旋转函数 (Rotate Function)
    public int maxRotateFunction(int[] nums) {
        int res = 0;
        int sum = 0;
        for (int i = 0; i < nums.length; ++i) {
            res += nums[i] * i;
            sum += nums[i];
        }
        int cur = res;
        for (int i = 1; i < nums.length; ++i) {
            cur += sum - nums[nums.length - i] * nums.length;
            res = Math.max(cur, res);
        }
        return res;

    }

    // 1203. 项目管理 (Sort Items by Groups Respecting Dependencies) --拓扑排序
    public int[] sortItems(int n, int m, int[] group, List<List<Integer>> beforeItems) {
        // 为 group[i] == -1 的组编号
        for (int i = 0; i < group.length; ++i) {
            if (group[i] == -1) {
                group[i] = m++;
            }
        }
        // 初始化group、item 邻接表
        List<Integer>[] groupsAdj = new ArrayList[m];
        List<Integer>[] itemsAdj = new ArrayList[n];
        for (int i = 0; i < m; ++i) {
            groupsAdj[i] = new ArrayList<>();
        }
        for (int i = 0; i < n; ++i) {
            itemsAdj[i] = new ArrayList<>();
        }
        // 设置邻接表和入度数组
        int[] groupsInDegrees = new int[m];
        int[] itemsInDegrees = new int[n];
        for (int i = 0; i < group.length; ++i) {
            int currentGroup = group[i];
            for (int beforeItem : beforeItems.get(i)) {
                int beforeGroup = group[beforeItem];
                if (beforeGroup != currentGroup) {
                    groupsAdj[beforeGroup].add(currentGroup);
                    ++groupsInDegrees[currentGroup];
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            for (int beforeItem : beforeItems.get(i)) {
                itemsAdj[beforeItem].add(i);
                ++itemsInDegrees[i];
            }
        }
        // 拓扑排序
        List<Integer> groupsList = topologicalSort(groupsAdj, groupsInDegrees);
        if (groupsList.isEmpty()) {
            return new int[0];
        }
        List<Integer> itemsList = topologicalSort(itemsAdj, itemsInDegrees);
        if (itemsList.isEmpty()) {
            return new int[0];
        }
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int item : itemsList) {
            map.computeIfAbsent(group[item], k -> new ArrayList<>()).add(item);
        }
        List<Integer> res = new ArrayList<>();
        for (int g : groupsList) {
            List<Integer> items = map.getOrDefault(g, new ArrayList<>());
            res.addAll(items);
        }
        return res.stream().mapToInt(Integer::valueOf).toArray();

    }

    private List<Integer> topologicalSort(List<Integer>[] listAdj, int[] inDegrees) {
        int n = inDegrees.length;
        Queue<Integer> queue = new LinkedList<>();
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            if (inDegrees[i] == 0) {
                queue.offer(i);
            }
        }
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            res.add(cur);
            for (int neighbor : listAdj[cur]) {
                if (--inDegrees[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        if (res.size() == n) {
            return res;
        }
        return new ArrayList<>();
    }

    // 743. 网络延迟时间 (Network Delay Time)
    public int networkDelayTime(int[][] times, int n, int k) {
        List<int[]>[] g = new ArrayList[n];
        for (int i = 0; i < n; ++i) {
            g[i] = new ArrayList<>();
        }
        for (int[] t : times) {
            int a = t[0] - 1;
            int b = t[1] - 1;
            int d = t[2];
            g[a].add(new int[] { b, d });
        }
        int[] dis = new int[n];
        Arrays.fill(dis, Integer.MAX_VALUE);
        dis[k - 1] = 0;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[1], o2[1]);
            }

        });

        q.offer(new int[] { k - 1, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int d = cur[1];
            if (d > dis[x]) {
                continue;
            }
            for (int[] nei : g[x]) {
                int y = nei[0];
                int delta = nei[1];
                if (d + delta < dis[y]) {
                    dis[y] = d + delta;
                    q.offer(new int[] { y, dis[y] });
                }
            }
        }
        int max = 0;
        for (int i = 0; i < n; ++i) {
            max = Math.max(max, dis[i]);
        }
        return max == Integer.MAX_VALUE ? -1 : max;

    }

    // 1334. 阈值距离内邻居最少的城市 (Find the City With the Smallest Number of Neighbors at a
    // Threshold Distance) --Dijkstra
    public int findTheCity(int n, int[][] edges, int distanceThreshold) {
        List<int[]>[] g = new ArrayList[n];
        for (int i = 0; i < n; ++i) {
            g[i] = new ArrayList<>();
        }
        for (int[] e : edges) {
            int a = e[0];
            int b = e[1];
            int d = e[2];
            g[a].add(new int[] { b, d });
            g[b].add(new int[] { a, d });
        }
        int minCities = n + 1;
        int res = -1;
        for (int i = n - 1; i >= 0; --i) {
            int curCities = getDis(g, n, i, distanceThreshold);
            if (curCities < minCities) {
                minCities = curCities;
                res = i;
            }
        }
        return res;

    }

    private int getDis(List<int[]>[] g, int n, int start, int distanceThreshold) {
        int count = 0;
        int[] dis = new int[n];
        Arrays.fill(dis, Integer.MAX_VALUE);
        dis[start] = 0;
        Queue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[1], o2[1]);
            }
        });

        q.offer(new int[] { start, 0 });
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            int x = cur[0];
            int d = cur[1];
            if (d > dis[x]) {
                continue;
            }
            for (int[] nei : g[x]) {
                int y = nei[0];
                int delta = nei[1];
                if (d + delta < dis[y]) {
                    dis[y] = d + delta;
                    q.offer(new int[] { y, dis[y] });
                }
            }
        }
        for (int i = 0; i < n; ++i) {
            if (i != start) {
                if (dis[i] <= distanceThreshold) {
                    ++count;
                }

            }
        }
        return count;

    }

    // 1404. 将二进制表示减到 1 的步骤数 (Number of Steps to Reduce a Number in Binary
    // Representation to One) --模拟
    public int numSteps(String s) {
        int res = 0;
        while (!"1".equals(s)) {
            ++res;
            if (s.charAt(s.length() - 1) == '0') {
                s = s.substring(0, s.length() - 1);
            } else {
                char[] chars = s.toCharArray();
                for (int i = chars.length - 1; i >= 0; --i) {
                    if (chars[i] == '1') {
                        chars[i] = '0';
                        if (i == 0) {
                            s = "1" + String.valueOf(chars);
                            break;
                        }
                    } else {
                        chars[i] = '1';
                        s = String.valueOf(chars);
                        break;
                    }
                }
            }
        }
        return res;
    }

    // 1404. 将二进制表示减到 1 的步骤数 (Number of Steps to Reduce a Number in Binary
    // Representation to One) -- 贪心
    public int numSteps2(String s) {
        boolean meet1 = false;
        int n = s.length();
        int res = 0;
        for (int i = n - 1; i >= 0; --i) {
            if (s.charAt(i) == '0') {
                res += meet1 ? 2 : 1;
            } else {
                if (!meet1) {
                    if (i != 0) {
                        res += 2;
                    }
                    meet1 = true;
                } else {
                    ++res;
                }
            }
        }
        return res;

    }

    // 514. 自由之路 (Freedom Trail)
    private int[][] memo514;
    private int n514;
    private int m514;
    private String ring514;
    private String key514;
    private List<Integer>[] map514;

    public int findRotateSteps(String ring, String key) {
        this.n514 = ring.length();
        this.map514 = new ArrayList[26];
        Arrays.setAll(map514, k -> new ArrayList<>());
        for (int i = 0; i < n514; ++i) {
            int index = ring.charAt(i) - 'a';
            map514[index].add(i);
        }
        this.key514 = key;
        this.ring514 = ring;
        this.m514 = key.length();
        this.memo514 = new int[m514][n514];
        for (int i = 0; i < m514; ++i) {
            Arrays.fill(memo514[i], -1);
        }
        return dfs514(0, 0);

    }

    private int dfs514(int i, int j) {
        if (i == m514) {
            return 0;
        }
        if (memo514[i][j] != -1) {
            return memo514[i][j];
        }
        if (key514.charAt(i) == ring514.charAt(j)) {
            return memo514[i][j] = dfs514(i + 1, j) + 1;
        }
        int res = Integer.MAX_VALUE;
        for (int id : map514[key514.charAt(i) - 'a']) {
            int step = Math.min(Math.abs(id - j) + 1, n514 - Math.abs(id - j) + 1);
            res = Math.min(res, dfs514(i + 1, id) + step);
        }
        return memo514[i][j] = res;
    }

    // 685. 冗余连接 II (Redundant Connection II) --dfs
    public int[] findRedundantDirectedConnection(int[][] edges) {
        int n = edges.length;
        for (int i = n - 1; i >= 0; --i) {
            if (check685(i, edges)) {
                return edges[i];
            }
        }
        return new int[0];
    }

    private boolean check685(int i, int[][] edges) {
        int n = edges.length;
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        int[] deg = new int[n];
        for (int j = 0; j < edges.length; ++j) {
            if (i == j) {
                continue;
            }
            g[edges[j][0] - 1].add(edges[j][1] - 1);
            ++deg[edges[j][1] - 1];
        }
        int root = -1;
        for (int j = 0; j < n; ++j) {
            if (deg[j] == 0) {
                root = j;
                break;
            }
        }
        boolean[] vis = new boolean[n];
        boolean res = dfs685(root, -1, vis, g);
        if (!res) {
            return false;
        }
        for (boolean v : vis) {
            if (!v) {
                return false;
            }
        }
        return true;
    }

    private boolean dfs685(int x, int fa, boolean[] vis, List<Integer>[] g) {
        vis[x] = true;
        for (int y : g[x]) {
            if (y != fa) {
                if (vis[y]) {
                    return false;
                }
                if (!dfs685(y, x, vis, g)) {
                    return false;
                }
            }
        }
        return true;
    }

    // 685. 冗余连接 II (Redundant Connection II) --并查集
    public int[] findRedundantDirectedConnection2(int[][] edges) {
        int n = edges.length;
        // 预处理
        for (int[] edge : edges) {
            --edge[0];
            --edge[1];
        }
        int[] inDegrees = new int[n];
        int nodeIndexWithTwoIndegrees = -1;
        for (int[] edge : edges) {
            ++inDegrees[edge[1]];
            if (inDegrees[edge[1]] == 2) {
                nodeIndexWithTwoIndegrees = edge[1];
            }
        }
        // 存在入度为2的节点
        if (nodeIndexWithTwoIndegrees != -1) {
            List<Integer> pos = new ArrayList<>();
            for (int i = edges.length - 1; i >= 0; --i) {
                if (edges[i][1] == nodeIndexWithTwoIndegrees) {
                    pos.add(i);
                }
            }
            // 判断删除一条边 能否成为一棵树🌲
            if (isTreeAfterRemoveEdge(edges, pos.get(0))) {
                int[] res = edges[pos.get(0)];
                ++res[0];
                ++res[1];
                return res;
            } else {
                int[] res = edges[pos.get(1)];
                ++res[0];
                ++res[1];
                return edges[pos.get(1)];
            }
        }
        // 存在环，删除构成环的边
        return deleteAEdgeToEliminateCycle(edges);

    }

    private boolean isTreeAfterRemoveEdge(int[][] edges, int removedEdge) {
        int n = edges.length;
        Union685 union = new Union685(n);
        for (int i = 0; i < n; ++i) {
            if (i != removedEdge) {
                if (union.isConnected(edges[i][0], edges[i][1])) {
                    return false;
                }
                union.union(edges[i][0], edges[i][1]);
            }
        }
        return true;
    }

    private int[] deleteAEdgeToEliminateCycle(int[][] edges) {
        int n = edges.length;
        Union685 union = new Union685(n);
        for (int i = 0; i < n; ++i) {
            if (union.isConnected(edges[i][0], edges[i][1])) {
                int[] res = edges[i];
                ++res[0];
                ++res[1];
                return res;
            }
            union.union(edges[i][0], edges[i][1]);
        }
        return null;
    }

    class Union685 {
        private int[] rank;
        private int[] parent;

        public Union685(int n) {
            this.rank = new int[n];
            Arrays.fill(rank, 1);
            this.parent = new int[n];
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

    // 854. 相似度为 K 的字符串 (K-Similar Strings) -- bfs
    public int kSimilarity(String s1, String s2) {
        Queue<String> queue = new LinkedList<>();
        queue.offer(s1);
        Map<String, Integer> visited = new HashMap<>();
        visited.put(s1, 0);
        while (!queue.isEmpty()) {
            String cur = queue.poll();
            if (cur.equals(s2)) {
                return visited.get(cur);
            }
            for (String neighbor : getNeighbor854(cur, s2)) {
                if (!visited.containsKey(neighbor)) {
                    visited.put(neighbor, visited.get(cur) + 1);
                    queue.offer(neighbor);
                }
            }
        }
        return 0;

    }

    private List<String> getNeighbor854(String s, String target) {
        int n = s.length();
        int i = 0;
        while (i < n) {
            if (s.charAt(i) != target.charAt(i)) {
                break;
            }
            ++i;
        }
        List<String> res = new ArrayList<>();
        char[] chars = s.toCharArray();
        for (int j = i + 1; j < n; ++j) {
            if (s.charAt(j) == target.charAt(i)) {
                swap854(chars, j, i);
                res.add(new String(chars));
                swap854(chars, j, i);
            }
        }
        return res;
    }

    private void swap854(char[] chars, int i, int j) {
        char temp = chars[i];
        chars[i] = chars[j];
        chars[j] = temp;
    }

    // 1361. 验证二叉树 (Validate Binary Tree Nodes) --拓扑排序
    public boolean validateBinaryTreeNodes(int n, int[] leftChild, int[] rightChild) {
        int[] inDegrees = new int[n];
        // 入度
        for (int left : leftChild) {
            if (left != -1) {
                ++inDegrees[left];
            }
        }
        for (int right : rightChild) {
            if (right != -1) {
                ++inDegrees[right];
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < inDegrees.length; ++i) {
            // 存在 入度 > 1 的节点
            if (inDegrees[i] > 1) {
                return false;
            }
            if (inDegrees[i] == 0) {
                queue.offer(i);
            }
        }
        // 树🌲：入度 == 0 的节点个数，有且只能有1个 (root)
        if (queue.size() != 1) {
            return false;
        }
        // 拓扑排序
        int count = 0;
        while (!queue.isEmpty()) {
            ++count;
            int cur = queue.poll();
            if (leftChild[cur] != -1) {
                --inDegrees[leftChild[cur]];
                if (inDegrees[leftChild[cur]] == 0) {
                    queue.offer(leftChild[cur]);
                }
            }

            if (rightChild[cur] != -1) {
                --inDegrees[rightChild[cur]];
                if (inDegrees[rightChild[cur]] == 0) {
                    queue.offer(rightChild[cur]);
                }
            }
        }
        return count == n;

    }

    // 1591. 奇怪的打印机 II (Strange Printer II) --拓扑排序
    public boolean isPrintable(int[][] targetGrid) {
        int m = targetGrid.length;
        int n = targetGrid[0].length;
        // from -> to 是否访问过
        boolean[][] visited = new boolean[61][61];
        // 每种颜色 上下左右四个方向的边界
        int[][] border = new int[61][4];
        for (int i = 0; i < border.length; ++i) {
            Arrays.fill(border[i], -1);
        }
        // 入度
        int[] inDegrees = new int[61];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int color = targetGrid[i][j];
                // 上边界
                border[color][0] = border[color][0] == -1 ? i : Math.min(border[color][0], i);
                // 下边界
                border[color][1] = border[color][1] == -1 ? i : Math.max(border[color][1], i);
                // 左边界
                border[color][2] = border[color][2] == -1 ? j : Math.min(border[color][2], j);
                // 右边界
                border[color][3] = border[color][3] == -1 ? j : Math.max(border[color][3], j);
            }
        }
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int to = targetGrid[i][j];
                for (int from = 1; from <= 60; ++from) {
                    if (from != to && !visited[from][to] && border[from][0] <= i && border[from][1] >= i
                            && border[from][2] <= j && border[from][3] >= j) {
                        visited[from][to] = true;
                        graph.computeIfAbsent(from, k -> new LinkedList<>()).add(to);
                        ++inDegrees[to];
                    }
                }
            }
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 1; i <= 60; ++i) {
            if (inDegrees[i] == 0) {
                queue.offer(i);
            }
        }
        int count = 0;
        while (!queue.isEmpty()) {
            ++count;
            int color = queue.poll();
            if (graph.get(color) == null) {
                continue;
            }
            for (int neighbor : graph.get(color)) {
                if (--inDegrees[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        return count == 60;

    }

    // 面试题 08.02. 迷路的机器人 (Robot in a Grid LCCI) --bfs
    public List<List<Integer>> pathWithObstacles(int[][] obstacleGrid) {

        int m = obstacleGrid.length;
        int n = obstacleGrid[0].length;
        if (obstacleGrid[0][0] == 1 || obstacleGrid[m - 1][n - 1] == 1) {
            return new ArrayList<>();
        }
        int[][] directions = { { 0, 1 }, { 1, 0 } };
        // son , parent
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, null);
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(0);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int pos = queue.poll();
                if (pos / n == m - 1 && pos % n == n - 1) {
                    return getPath0802(map, n, pos);
                }
                for (int[] direction : directions) {
                    int nx = pos / n + direction[0];
                    int ny = pos % n + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && obstacleGrid[nx][ny] != 1) {
                        map.put(getIndex0802(n, nx, ny), pos);
                        obstacleGrid[nx][ny] = 1;
                        queue.offer(getIndex0802(n, nx, ny));
                    }
                }
            }
        }
        return new ArrayList<>();
    }

    private List<List<Integer>> getPath0802(Map<Integer, Integer> map, int n, Integer pos) {
        List<List<Integer>> res = new ArrayList<>();
        do {
            List<Integer> next = new ArrayList<>();
            next.add(pos / n);
            next.add(pos % n);
            res.add(next);
            pos = map.get(pos);
        } while (pos != null);
        Collections.reverse(res);
        return res;
    }

    private int getIndex0802(int colCounts, int x, int y) {
        return x * colCounts + y;
    }

    // 2045. 到达目的地的第二短时间 (Second Minimum Time to Reach Destination) --bfs
    public int secondMinimum(int n, int[][] edges, int time, int change) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 1; i <= n; ++i) {
            map.put(i, new ArrayList<>());
        }
        for (int[] edge : edges) {
            map.get(edge[0]).add(edge[1]);
            map.get(edge[1]).add(edge[0]);
        }
        Queue<int[]> queue = new LinkedList<>();
        // 节点索引 , 距离节点1的距离
        queue.offer(new int[] { 1, 0 });
        // distance[i][0] : 节点1到节点i的最短距离
        // distance[i][1] : 节点1到节点i的次短距离
        int[][] distance = new int[n + 1][2];
        for (int i = 0; i < distance.length; ++i) {
            Arrays.fill(distance[i], Integer.MAX_VALUE);
        }
        distance[1][0] = 0;
        while (distance[n][1] == Integer.MAX_VALUE) {
            int[] cur = queue.poll();
            int node = cur[0];
            int dist = cur[1];
            int nDist = dist + 1;
            for (int neighbor : map.get(node)) {
                if (nDist < distance[neighbor][0]) {
                    distance[neighbor][0] = nDist;
                    queue.offer(new int[] { neighbor, nDist });
                } else if (nDist > distance[neighbor][0] && nDist < distance[neighbor][1]) {
                    distance[neighbor][1] = nDist;
                    queue.offer(new int[] { neighbor, nDist });
                }
            }
        }
        int res = 0;
        for (int i = 0; i < distance[n][1]; ++i) {
            if (res % (2 * change) >= change) {
                res = res + (2 * change - res % (2 * change));
            }
            res += time;
        }
        return res;

    }

    // 面试题 01.05. 一次编辑 (One Away LCCI)
    public boolean oneEditAway(String first, String second) {
        // 长度差最多为1
        int diff = Math.abs(first.length() - second.length());
        if (diff >= 2) {
            return false;
        }
        if (diff == 0) {
            return diffAtMostOneBit(first, second);
        }
        return insertOnlyOneBit(first, second);
    }

    private boolean insertOnlyOneBit(String s, String t) {
        if (s.length() > t.length()) {
            String temp = s;
            s = t;
            t = temp;
        }
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

    private boolean diffAtMostOneBit(String first, String second) {
        int diff = 0;
        for (int i = 0; i < first.length(); ++i) {
            if (first.charAt(i) != second.charAt(i)) {
                if (++diff > 1) {
                    return false;
                }
            }
        }
        return true;
    }

    // 161. 相隔为 1 的编辑距离 (One Edit Distance)
    public boolean isOneEditDistance(String s, String t) {
        if (Math.abs(s.length() - t.length()) > 1) {
            return false;
        }
        if (s.length() == t.length()) {
            return replaceOneToEqual(s, t);
        }
        return s.length() < t.length() ? addOneToEqual(s, t) : addOneToEqual(t, s);
    }

    private boolean addOneToEqual(String s, String t) {
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

    private boolean replaceOneToEqual(String s, String t) {
        int diff = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) != t.charAt(i)) {
                if (++diff > 1) {
                    return false;
                }
            }
        }
        return diff == 1;
    }

    // 161. 相隔为 1 的编辑距离 (One Edit Distance)
    public boolean isOneEditDistance2(String s, String t) {
        int ns = s.length();
        int nt = t.length();

        // Ensure that s is shorter than t.
        if (ns > nt)
            return isOneEditDistance(t, s);

        // The strings are NOT one edit away distance
        // if the length diff is more than 1.
        if (nt - ns > 1)
            return false;

        for (int i = 0; i < ns; i++) {
            if (s.charAt(i) != t.charAt(i)) {
                // if strings have the same length
                if (ns == nt) {
                    return s.substring(i + 1).equals(t.substring(i + 1));
                }

                // if strings have different lengths
                else {
                    return s.substring(i).equals(t.substring(i + 1));
                }
            }
        }
        // If there is no diffs on ns distance
        // the strings are one edit away only if
        // t has one more character.
        // "ab"、"abc"
        return (ns + 1 == nt);
    }

    // 1514. 概率最大的路径 (Path with Maximum Probability) --Dijkstra
    public double maxProbability(int n, int[][] edges, double[] succProb, int start, int end) {
        List<double[]>[] g = new ArrayList[n];
        for (int i = 0; i < n; ++i) {
            g[i] = new ArrayList<>();
        }
        for (int i = 0; i < edges.length; ++i) {
            int a = edges[i][0];
            int b = edges[i][1];
            double p = succProb[i];
            g[a].add(new double[] { b, p });
            g[b].add(new double[] { a, p });
        }
        double[] dis = new double[n];
        dis[start] = 1d;
        Queue<double[]> q = new PriorityQueue<>(new Comparator<double[]>() {

            @Override
            public int compare(double[] o1, double[] o2) {
                return Double.compare(o2[1], o1[1]);
            }

        });

        q.offer(new double[] { start, 1d });
        while (!q.isEmpty()) {
            double[] cur = q.poll();
            int x = (int) cur[0];
            double p = cur[1];
            if (Double.compare(p, dis[x]) < 0) {
                continue;
            }
            if (x == end) {
                break;
            }
            for (double[] nei : g[x]) {
                int y = (int) nei[0];
                double delta = nei[1];
                if (Double.compare(p * delta, dis[y]) > 0) {
                    dis[y] = p * delta;
                    q.offer(new double[] { y, dis[y] });

                }
            }
        }
        return dis[end];

    }

    // 1976. 到达目的地的方案数 (Number of Ways to Arrive at Destination) --Dijkstra
    public int countPaths(int n, int[][] roads) {
        List<int[]>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        for (int[] r : roads) {
            int a = r[0];
            int b = r[1];
            int t = r[2];
            g[a].add(new int[] { b, t });
            g[b].add(new int[] { a, t });
        }
        int[] dp = new int[n];
        final int MOD = (int) (1e9 + 7);
        long[] dis = new long[n];
        Arrays.fill(dis, Long.MAX_VALUE);
        dis[0] = 0L;
        dp[0] = 1;
        Queue<long[]> q = new PriorityQueue<>(new Comparator<long[]>() {

            @Override
            public int compare(long[] o1, long[] o2) {
                return Long.compare(o1[1], o2[1]);
            }

        });
        // node, time
        q.offer(new long[] { 0L, 0L });
        while (!q.isEmpty()) {
            long[] cur = q.poll();
            int x = (int) cur[0];
            long t = cur[1];
            for (int[] nei : g[x]) {
                int y = nei[0];
                int dt = nei[1];
                if (t + dt < dis[y]) {
                    dis[y] = t + dt;
                    dp[y] = dp[x];
                    q.offer(new long[] { y, t + dt });
                } else if (t + dt == dis[y]) {
                    dp[y] = (dp[y] + dp[x]) % MOD;
                }
            }
        }
        return dp[n - 1];

    }

    // 1046. 最后一块石头的重量 (Last Stone Weight)
    public int lastStoneWeight(int[] stones) {
        Queue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return o2.compareTo(o1);
            }

        });
        for (int stone : stones) {
            queue.offer(stone);
        }
        while (queue.size() > 1) {
            int first = queue.poll();
            int second = queue.poll();
            if (first != second) {
                queue.offer(first - second);
            }
        }
        return queue.isEmpty() ? 0 : queue.poll();

    }

    // 2255. 统计是给定字符串前缀的字符串数目 (Count Prefixes of a Given String)
    public int countPrefixes(String[] words, String s) {
        Trie2255 trie = new Trie2255();
        for (String word : words) {
            trie.insert(word);
        }
        return trie.countPrefixes(s);

    }

    public class Trie2255 {
        private Trie2255[] children;
        private int end;

        public Trie2255() {
            this.children = new Trie2255[26];
        }

        public void insert(String s) {
            Trie2255 node = this;
            for (char c : s.toCharArray()) {
                int idx = c - 'a';
                if (node.children[idx] == null) {
                    node.children[idx] = new Trie2255();
                }
                node = node.children[idx];
            }
            ++node.end;
        }

        public int countPrefixes(String s) {
            Trie2255 node = this;
            int cnt = 0;
            for (char c : s.toCharArray()) {
                int idx = c - 'a';
                if (node.children[idx] == null) {
                    break;
                }
                node = node.children[idx];
                cnt += node.end;
            }
            return cnt;
        }

    }

    // 6052. 最小平均差
    public int minimumAverageDifference(int[] nums) {
        if (nums.length == 1) {
            return 0;

        }
        int res = nums.length - 1;

        long sum = 0l;
        for (int num : nums) {
            sum += num;
        }
        long min = sum / nums.length;

        long leftSum = 0l;
        for (int i = 0; i < nums.length - 1; ++i) {
            leftSum += nums[i];
            long leftAverage = leftSum / (i + 1);

            long rightSum = sum - leftSum;
            long rightAverage = rightSum / (nums.length - i - 1);

            long diff = Math.abs(leftAverage - rightAverage);

            if (diff < min) {
                min = diff;
                res = i;
            } else if (diff == min) {
                res = Math.min(res, i);
            }
        }
        return res;

    }

    // 2257. 统计网格图中没有被保卫的格子数 (Count Unguarded Cells in the Grid)
    public int countUnguarded(int m, int n, int[][] guards, int[][] walls) {
        int[][] grid = new int[m][n];
        // guard = 1;
        // walls = 2;
        // watch = 3;
        for (int[] g : guards) {
            grid[g[0]][g[1]] = 1;
        }
        for (int[] w : walls) {
            grid[w[0]][w[1]] = 2;
        }
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };
        for (int[] g : guards) {
            for (int[] d : dirs) {
                int x = g[0] + d[0];
                int y = g[1] + d[1];
                while (x >= 0 && x < m && y >= 0 && y < n && (grid[x][y] == 0 || grid[x][y] == 3)) {
                    grid[x][y] = 3;
                    x += d[0];
                    y += d[1];
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0) {
                    ++res;
                }
            }
        }
        return res;

    }

    // 6054. 逃离火灾 (Escape the Spreading Fire) --bfs + 二分查找
    public int maximumMinutes(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = -1;
        int left = 0;
        int right = m * n - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check6054(mid, grid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (check6054(1000000000, grid)) {
            return 1000000000;
        }
        return res;

    }

    private boolean check6054(int time, int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        boolean[][] fire = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    fire[i][j] = true;
                    queue.offer(new int[] { i, j });
                }
            }
        }
        while (!queue.isEmpty()) {
            if (time == 0) {
                break;
            }
            --time;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = queue.poll();
                for (int[] direction : directions) {
                    int nx = cur[0] + direction[0];
                    int ny = cur[1] + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                        if (!fire[nx][ny] && grid[nx][ny] == 0) {
                            fire[nx][ny] = true;
                            queue.offer(new int[] { nx, ny });
                        }
                    }
                }
            }
        }
        if (fire[0][0] || fire[m - 1][n - 1]) {
            return false;
        }
        boolean[][] visited = new boolean[m][n];
        visited[0][0] = true;
        Queue<int[]> queuePerson = new LinkedList<>();
        queuePerson.offer(new int[] { 0, 0 });
        while (!queuePerson.isEmpty()) {
            int size = queuePerson.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = queuePerson.poll();
                int x = cur[0];
                int y = cur[1];
                if (!fire[x][y]) {
                    for (int[] direction : directions) {
                        int nx = x + direction[0];
                        int ny = y + direction[1];
                        if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                            if (!visited[nx][ny] && !fire[nx][ny] && grid[nx][ny] == 0) {
                                if (nx == m - 1 && ny == n - 1) {
                                    return true;
                                }
                                visited[nx][ny] = true;
                                queuePerson.offer(new int[] { nx, ny });
                            }
                        }
                    }
                }
            }
            size = queue.size();
            for (int i = 0; i < size; ++i) {
                int[] cur = queue.poll();
                int x = cur[0];
                int y = cur[1];

                for (int[] direction : directions) {
                    int nx = x + direction[0];
                    int ny = y + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                        if (!fire[nx][ny] && grid[nx][ny] == 0) {
                            fire[nx][ny] = true;
                            queue.offer(new int[] { nx, ny });
                        }
                    }
                }
            }
        }
        return false;
    }

    // 2259. 移除指定数字得到的最大结果 (Remove Digit From Number to Maximize Result)
    public String removeDigit(String number, char digit) {
        int firstIndex = number.indexOf(digit + "");
        String res = number.substring(0, firstIndex) + number.substring(firstIndex + 1);
        for (int i = firstIndex + 1; i < number.length(); ++i) {
            if (number.charAt(i) == digit) {
                String cur = number.substring(0, i) + number.substring(i + 1);
                if (cur.compareTo(res) > 0) {
                    res = cur;
                }
            }
        }
        return res;

    }

    // 2260. 必须拿起的最小连续卡牌数 --滑动窗口
    public int minimumCardPickup(int[] cards) {
        int left = 0;
        int right = 0;
        int res = Integer.MAX_VALUE;
        int[] counts = new int[1000001];
        while (right < cards.length) {
            ++counts[cards[right]];
            while (counts[cards[right]] >= 2) {
                res = Math.min(right - left + 1, res);
                if (res == 2) {
                    return res;
                }
                --counts[cards[left++]];
            }
            ++right;
        }
        return res == Integer.MAX_VALUE ? -1 : res;

    }

    // 2261. 含最多 K 个可整除元素的子数组 (K Divisible Elements Subarrays) --还需掌握字典树
    public int countDistinct(int[] nums, int k, int p) {
        int res = 0;
        Set<String> set = new HashSet<>();
        for (int i = 0; i < nums.length; ++i) {
            int count = 0;
            StringBuilder builder = new StringBuilder();
            for (int j = i; j >= 0; --j) {
                if (nums[j] % p == 0) {
                    ++count;
                }
                builder.append(nums[j] + " ");
                if (count <= k && !set.contains(builder.toString())) {
                    set.add(builder.toString());
                    ++res;
                }
            }
        }
        return res;

    }

    // 2262. 字符串的总引力 (Total Appeal of A String) --dp
    public long appealSum(String s) {
        int[] pos = new int[26];
        Arrays.fill(pos, -1);
        long sumG = 0l;
        long res = 0l;
        for (int i = 0; i < s.length(); ++i) {
            char c = s.charAt(i);
            sumG += i - pos[c - 'a'];
            res += sumG;
            pos[c - 'a'] = i;
        }
        return res;

    }

    // 463. 岛屿的周长 (Island Perimeter)
    public int islandPerimeter(int[][] grid) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int res = 0;
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    res += 4;
                    for (int[] direction : directions) {
                        int ni = i + direction[0];
                        int nj = j + direction[1];
                        if (ni >= 0 && ni < m && nj >= 0 && nj < n && grid[ni][nj] == 1) {
                            --res;
                        }
                    }
                }
            }
        }
        return res;
    }

    // 463. 岛屿的周长 (Island Perimeter) --bfs
    public int islandPerimeter2(int[][] grid) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        boolean[][] visited = new boolean[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    queue.offer(new int[] { i, j });
                    visited[i][j] = true;
                }
            }
        }
        int res = 0;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            res += 4;
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                    --res;
                    if (!visited[nx][ny]) {
                        visited[nx][ny] = true;
                        queue.offer(new int[] { nx, ny });
                    }
                }
            }
        }
        return res;

    }

    // 2264. 字符串中最大的 3 位相同数字 (Largest 3-Same-Digit Number in String)
    public String largestGoodInteger(String num) {
        String res = "";
        char[] arr = num.toCharArray();
        for (int i = 1; i < arr.length - 1; ++i) {
            if (arr[i] == arr[i - 1] && arr[i] == arr[i + 1]) {
                if (res.isEmpty() || res.compareTo(num.substring(i - 1, i + 2)) < 0) {
                    res = num.substring(i - 1, i + 2);
                }
            }
        }
        return res;

    }

    // 2269. 找到一个数字的 K 美丽值 (Find the K-Beauty of a Number)
    public int divisorSubstrings(int num, int k) {
        String s = String.valueOf(num);
        int res = 0;
        for (int i = k - 1; i < s.length(); ++i) {
            int cur = Integer.parseInt(s.substring(i - k + 1, i + 1));
            if (cur != 0 && num % cur == 0) {
                ++res;
            }
        }
        return res;

    }

    // 6067. 分割数组的方案数
    public int waysToSplitArray(int[] nums) {
        int res = 0;
        long sum = 0l;
        for (int num : nums) {
            sum += num;
        }
        long preSum = 0l;
        for (int i = 0; i < nums.length - 1; ++i) {
            preSum += nums[i];
            if (preSum >= sum - preSum) {
                ++res;
            }
        }
        return res;

    }

    // 2271. 毯子覆盖的最多白色砖块数 (Maximum White Tiles Covered by a Carpet) --贪心
    public static int maximumWhiteTiles(int[][] tiles, int carpetLen) {
        int res = 0;
        Arrays.sort(tiles, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });
        int min = tiles[0][0];
        for (int[] tile : tiles) {
            tile[0] -= min;
            tile[1] -= min;
        }
        int max = tiles[tiles.length - 1][1];
        int[] arr = new int[max + 1];
        int sum = 0;
        for (int[] tile : tiles) {
            Arrays.fill(arr, tile[0], tile[1] + 1, 1);
            sum += tile[1] - tile[0] + 1;
        }
        if (carpetLen >= max) {
            return sum;

        }
        int cur = 0;
        for (int i = 0; i < carpetLen; ++i) {
            cur += arr[i];
        }
        res = cur;
        for (int i = carpetLen; i < arr.length; ++i) {
            if (arr[i] == 1) {
                ++cur;
            }
            if (arr[i - carpetLen] == 1) {
                --cur;
            }
            res = Math.max(res, cur);
        }
        return res;

    }

    // 2271. 毯子覆盖的最多白色砖块数 (Maximum White Tiles Covered by a Carpet) --贪心
    public int maximumWhiteTiles2(int[][] tiles, int carpetLen) {
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

    // 2273. 移除字母异位词后的结果数组 (Find Resultant Array After Removing Anagrams)
    public List<String> removeAnagrams(String[] words) {
        List<String> res = new ArrayList<>();
        for (String w : words) {
            if (res.isEmpty() || !isAnagrams2273(res.get(res.size() - 1), w)) {
                res.add(w);
            }
        }
        return res;
    }

    // 是否异位词
    private boolean isAnagrams2273(String a, String b) {
        if (a.length() != b.length()) {
            return false;
        }
        int[] cnt = new int[26];
        for (char c : a.toCharArray()) {
            ++cnt[c - 'a'];
        }
        for (char c : b.toCharArray()) {
            if (--cnt[c - 'a'] < 0) {
                return false;
            }
        }
        return Arrays.equals(cnt, new int[26]);
    }

    // 6064. 不含特殊楼层的最大连续楼层数
    public int maxConsecutive(int bottom, int top, int[] special) {
        Arrays.sort(special);
        int res = Math.max(special[0] - bottom, top - special[special.length - 1]);

        for (int i = 1; i < special.length; ++i) {
            res = Math.max(res, special[i] - special[i - 1] - 1);
        }
        return res;

    }

    // 6065. 按位与结果大于零的最长组合
    public int largestCombination(int[] candidates) {
        int res = 0;
        for (int i = 0; i < 30; ++i) {
            int count = 0;
            for (int candidate : candidates) {
                if (((candidate >> i) & 1) == 1) {
                    ++count;
                }
            }
            res = Math.max(res, count);
        }
        return res;

    }

    // 2276. 统计区间中的整数数目 (Count Integers in Intervals) --柯朵莉树
    class CountIntervals {
        private TreeMap<Integer, Integer> map;
        private int cnt;

        public CountIntervals() {
            this.map = new TreeMap<>();
        }

        public void add(int left, int right) {
            while (map.ceilingKey(left - 1) != null && map.ceilingEntry(left - 1).getValue() - 1 <= right) {
                Integer key = map.ceilingKey(left - 1);
                left = Math.min(left, map.get(key));
                right = Math.max(right, key);
                cnt -= key - map.get(key) + 1;
                map.remove(key);
            }
            map.put(right, left);
            cnt += right - left + 1;
        }

        public int count() {
            return cnt;

        }
    }

}