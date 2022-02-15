import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class Leetcode_3 {
    public static void main(String[] args) {

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

    public class NodeDepth {
        public int depth;
        public TreeNode node;

        public NodeDepth(TreeNode node, int depth) {
            this.depth = depth;
            this.node = node;
        }

    }

    // 111. 二叉树的最小深度 (Minimum Depth of Binary Tree) --bfs
    public int minDepth(TreeNode root) {
        if (root == null) {
            return 0;
        }
        Queue<NodeDepth> queue = new LinkedList<>();
        queue.offer(new NodeDepth(root, 1));
        while (root != null) {
            NodeDepth nodeDepth = queue.poll();
            if (nodeDepth.node.left == null && nodeDepth.node.right == null) {
                return nodeDepth.depth;
            }
            if (nodeDepth.node.left != null) {
                queue.offer(new NodeDepth(nodeDepth.node.left, nodeDepth.depth + 1));
            }
            if (nodeDepth.node.right != null) {
                queue.offer(new NodeDepth(nodeDepth.node.right, nodeDepth.depth + 1));
            }
        }
        return 0;

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

}
