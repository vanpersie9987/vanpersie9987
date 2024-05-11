import java.util.ArrayDeque;
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
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Leetcode_9 {
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

    // 3008. 找出数组中的美丽下标 II (Find Beautiful Indices in the Given Array II)
    public List<Integer> beautifulIndices(String s, String a, String b, int k) {
        List<Integer> aList = zAlgorithm(a, s);
        List<Integer> bList = zAlgorithm(b, s);
        List<Integer> res = new ArrayList<>();
        int j = 0;
        for (int x : aList) {
            while (j < bList.size() && x - bList.get(j) > k) {
                ++j;
            }
            if (j < bList.size() && Math.abs(x - bList.get(j)) <= k) {
                res.add(x);
            }
        }
        return res;

    }

    private List<Integer> zAlgorithm(String t, String s) {
        String ss = t + s;
        int n = ss.length();
        int[] z = new int[n];
        int left = 0;
        int right = 0;
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i < n; ++i) {
            if (i <= right) {
                z[i] = Math.min(z[i - left], right - i + 1);
            }
            while (i + z[i] < n && ss.charAt(z[i]) == ss.charAt(i + z[i])) {
                left = i;
                right = i + z[i];
                ++z[i];
            }
            if (i >= t.length() && z[i] >= t.length()) {
                res.add(i - t.length());
            }
        }
        return res;
    }
}
