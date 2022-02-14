import java.util.LinkedList;
import java.util.Queue;

import LeetCodeText.Node;

public class Leetcode_3 {
    public static void main(String[] args) {

    }

    public class TreeNode {
        private int val;
        private TreeNode left;
        private TreeNode right;

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

}
