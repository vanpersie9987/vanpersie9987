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
import java.util.TreeSet;
import java.util.stream.Collectors;

public class Leetcode_5 {
    public static void main(String[] args) {
        // int[] nums1 = { -4, -4, 4, -1, -2, 5 };
        // int[] nums2 = { -2, 2, -1, 4, 4, 3 };
        // long res = numberOfPairs(nums1, nums2, 1);

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

    // 1451. 重新排列句子中的单词 (Rearrange Words in a Sentence)
    public String arrangeWords(String text) {
        StringBuilder builder = new StringBuilder(text);
        builder.replace(0, 1, String.valueOf(Character.toLowerCase(builder.charAt(0))));
        List<String> list = new ArrayList<>();
        int left = 0;
        int right = 0;
        while (right < builder.length()) {
            while (right < builder.length() && Character.isLetter(builder.charAt(right))) {
                ++right;
            }
            list.add(builder.substring(left, right));
            left = right + 1;
            right = left;
        }
        Collections.sort(list, new Comparator<String>() {

            @Override
            public int compare(String o1, String o2) {
                return o1.length() - o2.length();
            }

        });

        StringBuilder res = new StringBuilder();
        for (String s : list) {
            if (!res.isEmpty()) {
                res.append(" ");
            }
            res.append(s);
        }
        return res.replace(0, 1, String.valueOf(Character.toUpperCase(res.charAt(0)))).toString();
    }

    // 1845. 座位预约管理系统 (Seat Reservation Manager) --优先队列
    class SeatManager {
        private PriorityQueue<Integer> queue;

        public SeatManager(int n) {
            queue = new PriorityQueue<>();
            for (int i = 1; i <= n; ++i) {
                queue.offer(i);
            }
        }

        public int reserve() {
            return queue.poll();

        }

        public void unreserve(int seatNumber) {
            queue.offer(seatNumber);
        }
    }

    // 1749. 任意子数组和的绝对值的最大值 (Maximum Absolute Sum of Any Subarray)
    public int maxAbsoluteSum(int[] nums) {
        int prefix = 0;
        int min = 0;
        int max = 0;
        int res = 0;
        for (int num : nums) {
            prefix += num;
            res = Math.max(Math.abs(prefix - min), res);
            res = Math.max(Math.abs(prefix - max), res);
            min = Math.min(min, prefix);
            max = Math.max(max, prefix);
        }
        return res;

    }

    // 816. 模糊坐标 (Ambiguous Coordinates)
    public List<String> ambiguousCoordinates(String s) {
        int n = s.length();
        List<String> res = new ArrayList<>();
        for (int i = 2; i < n - 1; ++i) {
            for (String left : makeString816(s, 1, i)) {
                for (String right : makeString816(s, i, n - 1)) {
                    res.add("(" + left + ", " + right + ")");
                }
            }
        }
        return res;
    }

    private List<String> makeString816(String s, int i, int j) {
        List<String> list = new ArrayList<>();
        for (int d = 1; d <= j - i; ++d) {
            String left = s.substring(i, i + d);
            String right = s.substring(i + d, j);
            if ((!left.startsWith("0") || "0".equals(left)) && !right.endsWith("0")) {
                list.add(left + (d < j - i ? "." : "") + right);
            }
        }
        return list;
    }

    // 373. 查找和最小的 K 对数字 (Find K Pairs with Smallest Sums) --优先队列 + BFS 与786原理相同
    // （还需掌握二分查找）
    // 剑指 Offer II 061. 和最小的 k 个数对
    public List<List<Integer>> kSmallestPairs(int[] nums1, int[] nums2, int k) {
        int m = nums1.length;
        int n = nums2.length;
        PriorityQueue<int[]> priorityQueue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return nums1[o1[0]] + nums2[o1[1]] - (nums1[o2[0]] + nums2[o2[1]]);
            }

        });

        List<List<Integer>> res = new ArrayList<>();

        Set<String> visited = new HashSet<>();

        priorityQueue.offer(new int[] { 0, 0 });
        visited.add(0 + "_" + 0);
        while (k-- > 0 && !priorityQueue.isEmpty()) {
            int[] cur = priorityQueue.poll();
            int index1 = cur[0];
            int index2 = cur[1];
            res.add(List.of(nums1[index1], nums2[index2]));

            if (index1 + 1 < m && visited.add((index1 + 1) + "_" + index2)) {
                priorityQueue.offer(new int[] { index1 + 1, index2 });
            }

            if (index2 + 1 < n && visited.add(index1 + "_" + (index2 + 1))) {
                priorityQueue.offer(new int[] { index1, index2 + 1 });
            }
        }
        return res;

    }

    // 813. 最大平均值和的分组 (Largest Sum of Averages) --dp
    public double largestSumOfAverages(int[] nums, int k) {
        // dp[i][j] : 前i个元素，分成j份，可得到的最大平均值
        int n = nums.length;
        double[][] dp = new double[n + 1][k + 1];
        double[] prefix = new double[n + 1];
        for (int i = 1; i < n + 1; ++i) {
            prefix[i] = prefix[i - 1] + nums[i - 1];
        }

        for (int i = 1; i < n + 1; ++i) {
            dp[i][1] = prefix[i] / i;
            for (int kk = 2; kk <= k && kk <= i; ++kk) {
                for (int j = 1; j < i; ++j) {
                    dp[i][kk] = Math.max(dp[i][kk], dp[j][kk - 1] + (prefix[i] - prefix[j]) / (i - j));
                }
            }
        }
        return dp[n][k];

    }

    // 563. 二叉树的坡度 (Binary Tree Tilt) --dfs
    private int res563;

    public int findTilt(TreeNode root) {
        dfs563(root);
        return res563;
    }

    private int dfs563(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int left = dfs563(node.left);
        int right = dfs563(node.right);
        res563 += Math.abs(left - right);
        return left + right + node.val;
    }

    // 2054. 两个最好的不重叠活动 (Two Best Non-Overlapping Events) --优先队列
    public int maxTwoEvents(int[][] events) {
        Arrays.sort(events, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });

        PriorityQueue<int[]> priorityQueue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });
        int max = 0;
        int res = 0;

        for (int[] event : events) {
            while (!priorityQueue.isEmpty() && priorityQueue.peek()[0] < event[0]) {
                int[] cur = priorityQueue.poll();
                max = Math.max(max, cur[1]);
            }
            res = Math.max(res, max + event[2]);
            priorityQueue.offer(new int[] { event[1], event[2] });
        }
        return res;
    }

    // 437. 路径总和 III (Path Sum III)
    // 剑指 Offer II 050. 向下的路径节点之和
    // 面试题 04.12. 求和路径 (Paths with Sum LCCI)
    private int res437;
    private Map<Long, Integer> map437;

    public int pathSum(TreeNode root, int targetSum) {
        map437 = new HashMap<>();
        map437.put(0L, 1);
        dfs437(root, targetSum, 0L);
        return res437;

    }

    private void dfs437(TreeNode node, int targetSum, long cur) {
        if (node == null) {
            return;
        }
        cur += node.val;
        res437 += map437.getOrDefault(cur - targetSum, 0);
        map437.put(cur, map437.getOrDefault(cur, 0) + 1);
        dfs437(node.left, targetSum, cur);
        dfs437(node.right, targetSum, cur);
        map437.put(cur, map437.getOrDefault(cur, 0) - 1);
    }

    // 467. 环绕字符串中唯一的子字符串 (Unique Substrings in Wraparound String)
    public int findSubstringInWraproundString(String p) {
        Map<Character, Integer> map = new HashMap<>();
        int count = 0;
        char pre = '?';
        int res = 0;
        for (char c : p.toCharArray()) {
            if (pre != '?' && ((char) ((pre - 'a' + 1) % 26 + 'a')) == c) {
                ++count;
            } else {
                count = 1;
            }
            if (count > map.getOrDefault(c, 0)) {
                res += count - map.getOrDefault(c, 0);
                map.put(c, count);
            }
            pre = c;
        }
        return res;

    }

    // 467. 环绕字符串中唯一的子字符串 (Unique Substrings in Wraparound String)
    public int findSubstringInWraproundString2(String p) {
        int[] max = new int[26];
        int count = 0;
        int pre = -1;
        int res = 0;
        for (char c : p.toCharArray()) {
            int cur = c - 'a';
            if (pre != -1 && (pre + 1) % 26 == cur) {
                ++count;
            } else {
                count = 1;
            }
            if (count > max[cur]) {
                res += count - max[cur];
                max[cur] = count;
            }
            pre = cur;
        }
        return res;

    }

    // 1953. 你可以工作的最大周数 (Maximum Number of Weeks for Which You Can Work)
    public long numberOfWeeks(int[] milestones) {
        int max = 0;
        long sum = 0l;
        for (int milestone : milestones) {
            sum += milestone;
            max = Math.max(max, milestone);
        }
        sum -= max;

        if (max > sum + 1) {
            return sum * 2 + 1;
        }
        return sum + max;

    }

    // 809. 情感丰富的文字 (Expressive Words)
    public int expressiveWords(String s, String[] words) {
        int res = 0;
        search: for (String word : words) {
            int left1 = 0;
            int left2 = 0;
            int right1 = 0;
            int right2 = 0;
            while (right1 < s.length() && right2 < word.length()) {
                if (s.charAt(left1) != word.charAt(left2)) {
                    continue search;
                }
                while (right1 < s.length() && s.charAt(right1) == s.charAt(left1)) {
                    ++right1;
                }

                while (right2 < word.length() && word.charAt(right2) == word.charAt(left2)) {
                    ++right2;
                }

                int count1 = right1 - left1;
                int count2 = right2 - left2;
                if (count1 < count2) {
                    continue search;
                }
                if (count1 >= 3 || count1 == count2) {
                    left1 = right1;
                    left2 = right2;
                } else {
                    continue search;
                }
            }
            if (right1 == s.length() && right2 == word.length()) {
                ++res;
            }
        }
        return res;
    }

    // 809. 情感丰富的文字 (Expressive Words)
    public int expressiveWords2(String s, String[] words) {
        int res = 0;

        String shrinkS = getShrink809(s);
        List<Integer> countS = getCount809(s);

        search: for (String word : words) {
            String shrinkWord = getShrink809(word);
            if (!shrinkS.equals(shrinkWord)) {
                continue search;
            }
            List<Integer> countWord = getCount809(word);
            for (int i = 0; i < countS.size(); ++i) {
                int count1 = countS.get(i);
                int count2 = countWord.get(i);
                if (count1 < count2 || (count1 < 3 && count1 != count2)) {
                    continue search;
                }
            }
            ++res;
        }
        return res;
    }

    private List<Integer> getCount809(String s) {
        List<Integer> list = new ArrayList<>();
        int left = 0;
        int right = 0;
        while (right < s.length()) {
            while (right < s.length() && s.charAt(left) == s.charAt(right)) {
                ++right;
            }
            list.add(right - left);
            left = right;
        }
        return list;
    }

    private String getShrink809(String s) {
        StringBuilder builder = new StringBuilder();
        char pre = '?';
        for (char c : s.toCharArray()) {
            if (pre != '?' && pre == c) {
                continue;
            }
            builder.append(c);
            pre = c;
        }
        return builder.toString();
    }

    // 1954. 收集足够苹果的最小花园周长 (Minimum Garden Perimeter to Collect Enough Apples)
    public long minimumPerimeter(long neededApples) {
        long n = -1l;
        long left = 1l;
        long right = 100000l;
        while (left <= right) {
            long mid = left + ((right - left) >>> 1);
            if (2 * mid * (mid + 1) * (2 * mid + 1) >= neededApples) {
                n = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return n * 8;

    }

    // 6160. 和有限的最长子序列 (Longest Subsequence With Limited Sum)
    public int[] answerQueries(int[] nums, int[] queries) {
        int n = nums.length;
        int m = queries.length;
        int[] res = new int[m];
        Arrays.sort(nums);
        for (int i = 0; i < m; ++i) {
            int sum = 0;
            int j = 0;
            while (j < n && sum + nums[j] <= queries[i]) {
                sum += nums[j++];
            }
            res[i] = j;
        }
        return res;
    }

    // 6160. 和有限的最长子序列 (Longest Subsequence With Limited Sum) --前缀和 + 二分查找
    public int[] answerQueries2(int[] nums, int[] queries) {
        int n = nums.length;
        int m = queries.length;
        Arrays.sort(nums);
        for (int i = 1; i < n; ++i) {
            nums[i] += nums[i - 1];
        }
        for (int i = 0; i < m; ++i) {
            queries[i] = binarySearch6160(nums, queries[i]);
        }
        return queries;
    }

    /**
     * @return 排序数组nums中，小于等于target的元素个数
     */
    private int binarySearch6160(int[] nums, int target) {
        int n = nums.length;
        if (target < nums[0]) {
            return 0;
        } else if (target > nums[n - 1]) {
            return n;
        }
        int res = -1;
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] <= target) {
                res = mid + 1;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 6161. 从字符串中移除星号 (Removing Stars From a String)
    public String removeStars(String s) {
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c != '*') {
                stack.push(c);
            } else {
                stack.pop();
            }
        }
        StringBuilder res = new StringBuilder();
        while (!stack.isEmpty()) {
            res.insert(0, stack.pop());
        }
        return res.toString();

    }

    // 6162. 收集垃圾的最少总时间 (Minimum Amount of Time to Collect Garbage)
    public int garbageCollection(String[] garbage, int[] travel) {
        int n = garbage.length;
        int[][] counts = new int[n][3];
        for (int i = 0; i < garbage.length; ++i) {
            for (char c : garbage[i].toCharArray()) {
                if (c == 'M') {
                    ++counts[i][0];
                } else if (c == 'P') {
                    ++counts[i][1];
                } else {
                    ++counts[i][2];
                }
            }
        }
        int counts1 = getGarbage(counts, travel, 0);
        int counts2 = getGarbage(counts, travel, 1);
        int counts3 = getGarbage(counts, travel, 2);
        return counts1 + counts2 + counts3;

    }

    private int getGarbage(int[][] counts, int[] travel, int kind) {
        int res = 0;
        int i = counts.length - 1;
        while (i >= 0) {
            if (counts[i][kind] != 0) {
                break;
            }
            --i;
        }
        for (int j = 0; j <= i; ++j) {
            res += counts[j][kind];
        }
        for (int j = 0; j < i; ++j) {
            res += travel[j];
        }
        return res;
    }

    // 1567. 乘积为正数的最长子数组长度 (Maximum Length of Subarray With Positive Product) --dp
    // 空间O(n)
    public int getMaxLen(int[] nums) {
        int n = nums.length;
        int[] positive = new int[n];
        int[] negative = new int[n];
        int res = 0;
        if (nums[0] > 0) {
            positive[0] = 1;
            res = 1;
        } else if (nums[0] < 0) {
            negative[0] = 1;
        }
        for (int i = 1; i < n; ++i) {
            if (nums[i] > 0) {
                positive[i] = positive[i - 1] + 1;
                negative[i] = negative[i - 1] == 0 ? 0 : negative[i - 1] + 1;
            } else if (nums[i] < 0) {
                negative[i] = positive[i - 1] + 1;
                positive[i] = negative[i - 1] == 0 ? 0 : negative[i - 1] + 1;
            } else {
                positive[i] = 0;
                negative[i] = 0;
            }
            res = Math.max(res, positive[i]);
        }
        return res;

    }

    // 1567. 乘积为正数的最长子数组长度 (Maximum Length of Subarray With Positive Product) --dp
    // 空间O(1)
    public int getMaxLen2(int[] nums) {
        int n = nums.length;
        int positive = 0;
        int negative = 0;
        int res = 0;
        if (nums[0] > 0) {
            positive = 1;
            res = 1;
        } else if (nums[0] < 0) {
            negative = 1;
        }
        for (int i = 1; i < n; ++i) {
            if (nums[i] > 0) {
                positive = positive + 1;
                negative = negative == 0 ? 0 : negative + 1;
            } else if (nums[i] < 0) {
                int temp = negative;
                negative = positive + 1;
                positive = temp == 0 ? 0 : temp + 1;
            } else {
                positive = 0;
                negative = 0;
            }
            res = Math.max(res, positive);
        }
        return res;

    }

    // 2170. 使数组变成交替数组的最少操作数 (Minimum Operations to Make the Array Alternating)
    public int minimumOperations(int[] nums) {
        int n = nums.length;
        if (n == 1) {
            return 0;
        }
        Map<Integer, Integer> odd = new HashMap<>();
        Map<Integer, Integer> even = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            if (i % 2 == 0) {
                even.put(nums[i], even.getOrDefault(nums[i], 0) + 1);
            } else {
                odd.put(nums[i], odd.getOrDefault(nums[i], 0) + 1);
            }
        }
        int maxEvenKey = -1;
        int maxEvenCount = 0;
        int secondEvenCount = 0;
        for (Map.Entry<Integer, Integer> entry : even.entrySet()) {
            if (entry.getValue() >= maxEvenCount) {
                maxEvenKey = entry.getKey();
                secondEvenCount = maxEvenCount;
                maxEvenCount = entry.getValue();
            } else if (entry.getValue() >= secondEvenCount) {
                secondEvenCount = entry.getValue();
            }
        }
        int maxOddCount = 0;
        int secondOddCount = 0;
        int maxOddKey = -1;
        for (Map.Entry<Integer, Integer> entry : odd.entrySet()) {
            if (entry.getValue() >= maxOddCount) {
                maxOddKey = entry.getKey();
                secondOddCount = maxOddCount;
                maxOddCount = entry.getValue();
            } else if (entry.getValue() >= secondOddCount) {
                secondOddCount = entry.getValue();
            }
        }

        if (maxEvenKey != maxOddKey) {
            return n - maxEvenCount - maxOddCount;
        }
        return n - Math.max(maxEvenCount + secondOddCount, secondEvenCount + maxOddCount);

    }

    // 777. 在LR字符串中交换相邻字符 (Swap Adjacent in LR String)
    public boolean canTransform(String start, String end) {
        int n = start.length();
        int index1 = 0;
        int index2 = 0;
        while (index1 < n || index2 < n) {
            while (index1 < n && start.charAt(index1) == 'X') {
                ++index1;
            }
            while (index2 < n && end.charAt(index2) == 'X') {
                ++index2;
            }
            if (index1 == n && index2 == n) {
                return true;
            }
            if (index1 == n || index2 == n) {
                return false;
            }

            // condition : index1 < n && index2 < n
            if (start.charAt(index1) != end.charAt(index2)) {
                return false;
            }
            if (start.charAt(index1) == 'R') {
                if (index1 > index2) {
                    return false;
                }
            } else {
                if (index1 < index2) {
                    return false;
                }
            }
            ++index1;
            ++index2;
        }
        return true;

    }

    // 2392. 给定条件下构造矩阵 (Build a Matrix With Conditions)
    public int[][] buildMatrix(int k, int[][] rowConditions, int[][] colConditions) {
        int[] row = getTopologicalSort2392(rowConditions, k);
        int[] col = getTopologicalSort2392(colConditions, k);
        if (row.length < k || col.length < k) {
            return new int[][] {};
        }
        int[] pos = new int[k];
        for (int i = 0; i < k; ++i) {
            pos[col[i]] = i;
        }
        int[][] res = new int[k][k];
        for (int i = 0; i < k; ++i) {
            res[i][pos[row[i]]] = row[i] + 1;
        }

        // int[][] res = new int[k][k];
        // Map<Integer, List<Integer>> map = new HashMap<>();
        // for (int i = 0; i < k; ++i) {
        // map.computeIfAbsent(row[i], o -> new ArrayList<>()).add(i);
        // }
        // for (int i = 0; i < k; ++i) {
        // map.computeIfAbsent(col[i], o -> new ArrayList<>()).add(i);
        // }

        // for (int key : map.keySet()) {
        // res[map.get(key).get(0)][map.get(key).get(1)] = key + 1;
        // }
        return res;

    }

    private int[] getTopologicalSort2392(int[][] conditions, int k) {
        Map<Integer, List<Integer>> graph = new HashMap<>();
        int[] inDegrees = new int[k];
        for (int[] condition : conditions) {
            graph.computeIfAbsent(condition[0] - 1, o -> new ArrayList<>()).add(condition[1] - 1);
            ++inDegrees[condition[1] - 1];
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < k; ++i) {
            if (inDegrees[i] == 0) {
                queue.offer(i);
            }
        }
        List<Integer> res = new ArrayList<>();
        while (!queue.isEmpty()) {
            int cur = queue.poll();
            res.add(cur);
            for (int neighbor : graph.getOrDefault(cur, new ArrayList<>())) {
                --inDegrees[neighbor];
                if (inDegrees[neighbor] == 0) {
                    queue.offer(neighbor);
                }
            }
        }
        return res.stream().mapToInt(o -> o).toArray();

    }

    // 1881. 插入后的最大值 (Maximum Value after Insertion)
    public String maxValue(String n, int x) {
        StringBuilder builder = new StringBuilder(n);
        if (n.charAt(0) == '-') {
            for (int i = 1; i < builder.length(); ++i) {
                if (builder.charAt(i) - '0' > x) {
                    builder.insert(i, x);
                    return builder.toString();
                }
            }

        } else {
            for (int i = 0; i < builder.length(); ++i) {
                if (builder.charAt(i) - '0' < x) {
                    builder.insert(i, x);
                    return builder.toString();
                }
            }
        }
        return builder.append(x).toString();

    }

    // 722. 删除注释 (Remove Comments)
    public List<String> removeComments(String[] source) {
        boolean inBlock = false;
        List<String> res = new ArrayList<>();
        StringBuilder newLine = new StringBuilder();
        for (String s : source) {
            char[] chars = s.toCharArray();
            int n = chars.length;
            int i = 0;
            if (!inBlock) {
                newLine = new StringBuilder();
            }
            while (i < n) {
                if (!inBlock && i + 1 < n && chars[i] == '/' && chars[i + 1] == '*') {
                    inBlock = true;
                    ++i;
                } else if (inBlock && i + 1 < n && chars[i] == '*' && chars[i + 1] == '/') {
                    inBlock = false;
                    ++i;
                } else if (!inBlock && i + 1 < n && chars[i] == '/' && chars[i + 1] == '/') {
                    break;
                } else if (!inBlock) {
                    newLine.append(chars[i]);
                }
                ++i;
            }
            if (!inBlock && !newLine.isEmpty()) {
                res.add(newLine.toString());
            }
        }
        return res;
    }

    // 673. 最长递增子序列的个数 (Number of Longest Increasing Subsequence) --参考300题 dp
    // 还需掌握 ：贪心 + 前缀和 + 二分查找
    public int findNumberOfLIS(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        int[] count = new int[n];
        int maxLen = 0;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            dp[i] = 1;
            count[i] = 1;
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    if (dp[j] + 1 > dp[i]) {
                        dp[i] = dp[j] + 1;
                        count[i] = count[j];
                    } else if (dp[j] + 1 == dp[i]) {
                        count[i] += count[j];
                    }
                }
            }

            if (dp[i] > maxLen) {
                maxLen = dp[i];
                res = count[i];
            } else if (dp[i] == maxLen) {
                res += count[i];
            }
        }
        return res;

    }

    // 793. 阶乘函数后 K 个零 (Preimage Size of Factorial Zeroes Function) --二分查找 参考172题
    public int preimageSizeFZF(int k) {
        return help793(k + 1) - help793(k);

    }

    private int help793(int n) {
        long left = 0l;
        long right = 5l * n;
        int res = -1;
        while (left <= right) {
            long mid = left + ((right - left) >>> 1);
            if (zeta(mid) < n) {
                left = mid + 1;
            } else {
                res = (int) mid;
                right = mid - 1;
            }
        }
        return res;

    }

    private int zeta(long n) {
        int res = 0;
        while (n > 0) {
            res += n / 5;
            n /= 5;
        }
        return res;
    }

    // 983. 最低票价 (Minimum Cost For Tickets) --dp
    public int mincostTickets(int[] days, int[] costs) {
        int n = days.length;
        int maxDay = days[n - 1];
        int minDay = days[0];
        // dp[day] 以 day 开始的最低票价
        int[] dp = new int[maxDay + 31];
        int i = n - 1;
        int d = maxDay;
        while (d >= minDay) {
            if (d == days[i]) {
                dp[d] = Math.min(dp[d + 1] + costs[0], dp[d + 7] + costs[1]);
                dp[d] = Math.min(dp[d], dp[d + 30] + costs[2]);
                --i;
                --d;
            } else {
                Arrays.fill(dp, days[i] + 1, days[i + 1], dp[d + 1]);
                d = days[i];
            }
        }
        return dp[minDay];

    }

    // This is the custom function interface.
    // You should not implement it, or speculate about its implementation
    interface CustomFunction {
        // Returns f(x, y) for any given positive integers x and y.
        // Note that f(x, y) is increasing with respect to both x and y.
        // i.e. f(x, y) < f(x + 1, y), f(x, y) < f(x, y + 1)
        public int f(int x, int y);
    };

    // 1237. 找出给定方程的正整数解 (Find Positive Integer Solution for a Given Equation)
    public List<List<Integer>> findSolution(CustomFunction customfunction, int z) {
        List<List<Integer>> res = new ArrayList<>();
        int i = 1;
        int j = 1000;
        while (i <= 1000 && j >= 1) {
            int candidate = customfunction.f(i, j);
            if (candidate == z) {
                res.add(List.of(i++, j--));
                continue;
            }
            if (candidate < z) {
                ++i;
            } else {
                --j;
            }
        }
        return res;

    }

    // 848. 字母移位 (Shifting Letters)
    public String shiftingLetters(String s, int[] shifts) {
        int n = s.length();
        for (int i = n - 2; i >= 0; --i) {
            shifts[i] = (shifts[i] + shifts[i + 1]) % 26;
        }
        char[] chars = s.toCharArray();
        for (int i = 0; i < n; ++i) {
            chars[i] = (char) ((chars[i] - 'a' + shifts[i]) % 26 + 'a');
        }
        return String.valueOf(chars);

    }

    // 831. 隐藏个人信息 (Masking Personal Information)
    public String maskPII(String s) {
        StringBuilder res = new StringBuilder();
        int atIndex = s.indexOf("@");
        // 邮箱
        if (s.contains("@")) {
            res.append(s.substring(0, 1)).append("*****").append(s.substring(atIndex - 1));
        } else {
            // 电话
            // String digits = S.replaceAll("\\D+", "");
            int count = 0;
            StringBuilder last = new StringBuilder();
            for (int i = s.length() - 1; i >= 0; --i) {
                if (Character.isDigit(s.charAt(i))) {
                    ++count;
                    if (count <= 4) {
                        last.insert(0, s.charAt(i));
                    }
                }
            }
            if (count > 10) {
                res.append("+");
                while (count > 10) {
                    res.append("*");
                    if (count == 11) {
                        res.append("-");
                    }
                    --count;
                }
            }
            res.append("***-***-");
            res.append(last);
        }
        return res.toString().toLowerCase();

    }

    // 754. 到达终点数字 (Reach a Number)
    public int reachNumber(int target) {
        target = Math.abs(target);
        int sum = 0;
        for (int i = 1; i <= Integer.MAX_VALUE; ++i) {
            sum += i;
            if (sum >= target && (sum - target) % 2 == 0) {
                return i;
            }
        }
        return 0;

    }

    // 2399. 检查相同字母间的距离 (Check Distances Between Same Letters)
    public boolean checkDistances(String s, int[] distance) {
        int[] diff = new int[26];
        Arrays.fill(diff, -1);
        for (int i = 0; i < s.length(); ++i) {
            int index = s.charAt(i) - 'a';
            if (diff[index] == -1) {
                diff[index] = i;
            } else {
                diff[index] = i - diff[index] - 1;
            }
        }
        for (int i = 0; i < 26; ++i) {
            if (diff[i] != -1) {
                if (distance[i] != diff[i]) {
                    return false;
                }
            }
        }
        return true;

    }

    // 2401. 最长优雅子数组 (Longest Nice Subarray) 时间：O(n) 空间：O(n)
    public int longestNiceSubarray(int[] nums) {
        int n = nums.length;
        int max = 0;
        for (int num : nums) {
            max = Math.max(max, num);
        }
        int[] counts = new int[Integer.toBinaryString(max).length() + 1];
        int res = 1;
        int left = 0;
        int right = 0;
        while (right < n) {
            addCounts6169(counts, nums[right]);
            while (getMaxCounts(counts) > 1) {
                mineCounts6169(counts, nums[left++]);
            }
            res = Math.max(res, right - left + 1);
            ++right;

        }
        return res;

    }

    private int getMaxCounts(int[] counts) {
        int max = 0;
        for (int count : counts) {
            max = Math.max(max, count);
        }
        return max;
    }

    private void mineCounts6169(int[] counts, int num) {
        int i = 0;
        while (num != 0) {
            if (num % 2 != 0) {
                --counts[i];
            }
            ++i;
            num /= 2;
        }
    }

    private void addCounts6169(int[] counts, int num) {
        int i = 0;
        while (num != 0) {
            if (num % 2 != 0) {
                ++counts[i];
            }
            ++i;
            num /= 2;
        }
    }

    // 2401. 最长优雅子数组 (Longest Nice Subarray) 时间：O(nlog(max(nums))) 空间：O(1)
    public int longestNiceSubarray2(int[] nums) {
        int res = 0;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            int or = 0;
            int j = i;
            while (j >= 0 && (or & nums[j]) == 0) {
                or |= nums[j--];
            }
            res = Math.max(res, i - j);
        }
        return res;

    }

    // 2401. 最长优雅子数组 (Longest Nice Subarray) 时间：O(n) 空间：O(1)
    public int longestNiceSubarray3(int[] nums) {
        int left = 0;
        int right = 0;
        int mask = 0;
        int res = 0;
        int n = nums.length;
        while (right < n) {
            while ((mask & nums[right]) != 0) {
                mask ^= nums[left++];
            }
            res = Math.max(res, right - left + 1);
            mask ^= nums[right++];
        }
        return res;

    }

    // 2395. 和相等的子数组
    public boolean findSubarrays(int[] nums) {
        int n = nums.length;
        Set<Integer> set = new HashSet<>();
        for (int i = 1; i < n; ++i) {
            int sum = nums[i - 1] + nums[i];
            if (set.contains(sum)) {
                return true;
            }
            set.add(sum);
        }
        return false;

    }

    // 2396. 严格回文的数字 (Strictly Palindromic Number)
    public boolean isStrictlyPalindromic(int n) {
        for (int i = 2; i <= n - 2; ++i) {
            if (!getXNary(n, i)) {
                return false;
            }
        }
        return true;

    }

    private boolean getXNary(int n, int i) {
        StringBuilder builder = new StringBuilder();
        while (n != 0) {
            builder.append(n % i);
            n /= i;
        }
        return builder.toString().equals(builder.reverse().toString());
    }

    // 2397. 被列覆盖的最多行数 (Maximum Rows Covered by Columns)
    public int maximumRows(int[][] matrix, int numSelect) {
        int m = matrix.length;
        int n = matrix[0].length;
        int max = 1 << n;
        int[] arr = new int[m];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                arr[i] |= matrix[i][j] << j;
            }
        }
        int res = 0;
        for (int mask = 0; mask < max; ++mask) {
            if (n - Integer.bitCount(mask) != numSelect) {
                continue;
            }
            int cur = 0;
            for (int k = 0; k < arr.length; ++k) {
                if ((arr[k] & mask) == 0) {
                    ++cur;
                }
            }
            res = Math.max(res, cur);
        }
        return res;

    }

    // 2398. 预算内的最多机器人数目 (Maximum Number of Robots Within Budget)
    // --同：239 双指针 + 单调队列 + 滑动窗口
    public int maximumRobots(int[] chargeTimes, int[] runningCosts, long budget) {
        int n = chargeTimes.length;
        Deque<Integer> deque = new LinkedList<>();
        int left = 0;
        int right = 0;
        long sum = 0l;
        int res = 0;
        while (right < n) {
            while (!deque.isEmpty() && chargeTimes[deque.peekLast()] <= chargeTimes[right]) {
                deque.pollLast();
            }
            deque.offerLast(right);
            sum += runningCosts[right];
            while (!deque.isEmpty() && chargeTimes[deque.peekFirst()] + (right - left + 1) * sum > budget) {
                if (deque.peekFirst() == left) {
                    deque.pollFirst();
                }
                sum -= runningCosts[left++];
            }
            res = Math.max(res, right - left + 1);
            ++right;
        }
        return res;

    }

    // 828. 统计子串中的唯一字符 (Count Unique Characters of All Substrings of a Given String)
    public int uniqueLetterString(String s) {
        int[] last = new int[26];
        Arrays.fill(last, -1);
        int n = s.length();
        // left[i] : s[i]左侧的第一个与s[i]相同的字符索引，不存在则为 -1
        int[] left = new int[n];
        for (int i = 0; i < n; ++i) {
            int index = s.charAt(i) - 'A';
            left[i] = last[index];
            last[index] = i;
        }

        Arrays.fill(last, n);
        // right[i] : s[i]右侧的第一个与s[i]相同的字符索引，不存在则为 n
        int[] right = new int[n];
        for (int i = n - 1; i >= 0; --i) {
            int index = s.charAt(i) - 'A';
            right[i] = last[index];
            last[index] = i;
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            res += (i - left[i]) * (right[i] - i);
        }
        return res;

    }

    // 1401. 圆和矩形是否有重叠 (Circle and Rectangle Overlapping)
    public boolean checkOverlap(int radius, int xCenter, int yCenter, int x1, int y1, int x2, int y2) {
        if (inRectangle1401(xCenter, yCenter, x1, y1, x2, y2)) {
            return true;
        }

        if (inRectangle1401(xCenter, yCenter, x1 - radius, y1, x1, y2)) {
            return true;
        }
        if (inRectangle1401(xCenter, yCenter, x1, y2, x2, y2 + radius)) {
            return true;
        }
        if (inRectangle1401(xCenter, yCenter, x2, y1, x2 + radius, y2)) {
            return true;
        }
        if (inRectangle1401(xCenter, yCenter, x1, y1 - radius, x2, y1)) {
            return true;
        }

        if (inCircle1401(xCenter, yCenter, x1, y1) <= radius * radius) {
            return true;
        }
        if (inCircle1401(xCenter, yCenter, x1, y2) <= radius * radius) {
            return true;
        }
        if (inCircle1401(xCenter, yCenter, x2, y2) <= radius * radius) {
            return true;
        }
        if (inCircle1401(xCenter, yCenter, x2, y1) <= radius * radius) {
            return true;
        }
        return false;

    }

    private int inCircle1401(int xCenter, int yCenter, int x1, int y1) {
        return (xCenter - x1) * (xCenter - x1) + (yCenter - y1) * (yCenter - y1);
    }

    private boolean inRectangle1401(int xCenter, int yCenter, int x1, int y1, int x2, int y2) {
        return xCenter >= x1 && xCenter <= x2 && yCenter >= y1 && yCenter <= y2;

    }

    // 246. 中心对称数 (Strobogrammatic Number) --plus
    public boolean isStrobogrammatic(String num) {
        char[] chars = num.toCharArray();
        int n = chars.length;
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            if (left == right) {
                if (!isLegal246(chars[left])) {
                    return false;
                }
                switch246(chars, left);
                break;
            }
            if (!isLegal246(chars[left]) || !isLegal246(chars[right])) {
                return false;
            }
            char temp = chars[left];
            chars[left] = chars[right];
            chars[right] = temp;
            switch246(chars, left);
            switch246(chars, right);
            ++left;
            --right;
        }
        return String.valueOf(chars).equals(num);

    }

    private void switch246(char[] chars, int index) {
        switch (chars[index]) {
            case '6':
                chars[index] = '9';
                break;
            case '9':
                chars[index] = '6';
                break;
            default:
                break;
        }
    }

    private boolean isLegal246(char c) {
        return c == '0' || c == '1' || c == '6' || c == '8' || c == '9';
    }

    // 246. 中心对称数 (Strobogrammatic Number) --plus
    public boolean isStrobogrammatic2(String num) {
        Map<Character, Character> map = new HashMap<>();
        map.put('6', '9');
        map.put('9', '6');
        map.put('8', '8');
        map.put('0', '0');
        map.put('1', '1');
        int left = 0;
        int right = num.length() - 1;
        while (left <= right) {
            if (!map.containsKey(num.charAt(left))
                    || !map.containsKey(num.charAt(right))
                    || map.get(num.charAt(left)) != num.charAt(right)) {
                return false;
            }
            ++left;
            --right;

        }
        return true;

    }

    // 252. 会议室 (Meeting Rooms) --plus
    public boolean canAttendMeetings(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });

        for (int i = 1; i < intervals.length; ++i) {
            int lastEnd = intervals[i - 1][1];
            int curStart = intervals[i][0];
            if (curStart < lastEnd) {
                return false;
            }
        }
        return true;

    }

    // 422. 有效的单词方块 (Valid Word Square) --plus
    public boolean validWordSquare(List<String> words) {
        int n = words.size();
        for (int i = 0; i < n; ++i) {
            String word = words.get(i);
            for (int j = 0; j < word.length(); ++j) {
                if (j >= n) {
                    return false;
                }
                if (words.get(j).length() <= i) {
                    return false;
                }
                if (word.charAt(j) != words.get(j).charAt(i)) {
                    return false;
                }
            }
        }
        return true;

    }

    // 2393. Count Strictly Increasing Subarrays --plus
    public long countSubarrays(int[] nums) {
        int count = 0;
        int pre = -1;
        long res = 0l;
        for (int num : nums) {
            if (num > pre) {
                ++count;
            } else {
                count = 1;
            }
            res += count;
            pre = num;
        }
        return res;

    }

    interface ArrayReader {
        public int get(int index);
    }

    // 702. 搜索长度未知的有序数组 (Search in a Sorted Array of Unknown Size) --plus
    public int search(ArrayReader reader, int target) {
        int left = 0;
        int right = 10001;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            int search = reader.get(mid);

            if (search == target) {
                return mid;
            } else if (search == Integer.MAX_VALUE || search > target) {
                right = mid - 1;
            } else if (search < target) {
                left = mid + 1;
            }
        }
        return -1;

    }

    // 2291. Maximum Profit From Trading Stocks --plus
    // -- 0-1背包（至多选择一次；外层循环：nums ； 内层循环：target；倒序遍历)
    // --求最值dp[i]=max/min(dp[i], dp[i-nums] + 1)或dp[i]=max/min(dp[i], dp[i-num] +
    // nums);
    public int maximumProfit(int[] present, int[] future, int budget) {
        int n = present.length;
        // dp[i] ： 预算为i时，可获得的最大利润
        int[] dp = new int[budget + 1];
        for (int i = 0; i < n; ++i) {
            if (future[i] > present[i]) {
                for (int j = budget; j >= present[i]; --j) {
                    dp[j] = Math.max(dp[j], dp[j - present[i]] + future[i] - present[i]);
                }
            }
        }
        return dp[budget];

    }

    interface Relation {
        boolean knows(int a, int b);
    }

    // 277. 搜寻名人 (Find the Celebrity) --plus
    class Resolusion277 implements Relation {

        public int findCelebrity(int n) {
            int i = 0;
            int j = 1;
            while (j < n) {
                if (knows(i, j)) {
                    i = j;
                }
                ++j;
            }
            while (--j > i) {
                if (!knows(j, i)) {
                    return -1;
                }
            }
            while (--j >= 0) {
                if (!knows(j, i) || knows(i, j)) {
                    return -1;
                }
            }
            return i;
        }

        @Override
        public boolean knows(int a, int b) {
            return false;
        }

    }

    // 2361. 乘坐火车路线的最少费用 (Minimum Costs Using the Train Line) --plus
    public long[] minimumCosts(int[] regular, int[] express, int expressCost) {
        int n = regular.length;
        long[] res = new long[n];
        // dp[i][0] ：从第 0 站到第 i 的普通站 所需的最小费用
        // dp[i][1] ：从第 0 站到第 i 的特快站 所需的最小费用
        long[][] dp = new long[n + 1][2];
        dp[0][1] = expressCost;
        for (int i = 0; i < n; ++i) {
            dp[i + 1][0] = Math.min(dp[i][0] + regular[i], dp[i][1] + express[i]);
            dp[i + 1][1] = Math.min(dp[i][0] + regular[i] + expressCost, dp[i][1] + express[i]);
            res[i] = Math.min(dp[i + 1][0], dp[i + 1][1]);
        }
        return res;

    }

    // 2361. 乘坐火车路线的最少费用 (Minimum Costs Using the Train Line) --plus
    public long[] minimumCosts2(int[] regular, int[] express, int expressCost) {
        int n = regular.length;
        long[] res = new long[n];
        long reg = 0;
        long exp = expressCost;
        for (int i = 0; i < n; ++i) {
            long tempReg = reg;
            reg = Math.min(reg + regular[i], exp + express[i]);
            exp = Math.min(tempReg + regular[i] + expressCost, exp + express[i]);
            res[i] = Math.min(reg, exp);
        }
        return res;

    }

    // 6176. 出现最频繁的偶数元素
    public int mostFrequentEven(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (num % 2 == 0) {
                map.put(num, map.getOrDefault(num, 0) + 1);
            }
        }
        if (map.isEmpty()) {
            return -1;
        }
        int max = Collections.max(map.values());
        int min = Integer.MAX_VALUE;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (max == entry.getValue()) {
                if (min > entry.getKey()) {
                    min = entry.getKey();
                }
            }
        }
        return min;

    }

    // 6177. 子字符串的最优划分
    public int partitionString(String s) {
        int res = 0;
        int mask = 0;
        for (char c : s.toCharArray()) {
            if ((mask | (1 << (c - 'a'))) == mask) {
                ++res;
                mask = 0;
            }
            mask |= 1 << (c - 'a');
        }
        return res + 1;

    }

    // 6178. 将区间分为最少组数
    public int minGroups(int[][] intervals) {
        int max = 0;
        for (int[] interval : intervals) {
            max = Math.max(max, interval[1]);
        }
        int[] diff = new int[max + 2];
        for (int[] interval : intervals) {
            ++diff[interval[0]];
            --diff[interval[1] + 1];
        }
        int res = diff[0];
        for (int i = 1; i < diff.length; ++i) {
            diff[i] += diff[i - 1];
            res = Math.max(res, diff[i]);
        }
        return res;

    }

    // 1387. 将整数按权重排序 (Sort Integers by The Power Value) --plus
    public int getKth(int lo, int hi, int k) {
        List<Integer> list = new ArrayList<>();
        for (int i = lo; i <= hi; ++i) {
            list.add(i);
        }
        Collections.sort(list, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                int p1 = getPowerValue1387(o1);
                int p2 = getPowerValue1387(o2);
                if (p1 == p2) {
                    return o1 - o2;
                }
                return p1 - p2;
            }

            private int getPowerValue1387(int num) {
                int step = 0;
                while (num != 1) {
                    if (num % 2 == 0) {
                        num /= 2;
                    } else {
                        num = 3 * num + 1;
                    }
                    ++step;
                }
                return step;
            }

        });

        return list.get(k - 1);

    }

    // 1387. 将整数按权重排序 (Sort Integers by The Power Value) --plus 记忆化搜索
    private Map<Integer, Integer> map1387;

    public int getKth2(int lo, int hi, int k) {
        map1387 = new HashMap<>();
        List<Integer> list = new ArrayList<>();
        for (int i = lo; i <= hi; ++i) {
            list.add(i);
        }
        Collections.sort(list, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                int p1 = getPowerValue1387(o1);
                int p2 = getPowerValue1387(o2);
                if (p1 == p2) {
                    return o1 - o2;
                }
                return p1 - p2;

            }

            private int getPowerValue1387(int num) {
                if (!map1387.containsKey(num)) {
                    if (num == 1) {
                        map1387.put(num, 0);
                    } else if (num % 2 == 0) {
                        map1387.put(num, getPowerValue1387(num / 2) + 1);
                    } else {
                        map1387.put(num, getPowerValue1387(3 * num + 1) + 1);
                    }
                }
                return map1387.get(num);
            }

        });
        return list.get(k - 1);

    }

    // 323. 无向图中连通分量的数目 (Number of Connected Components in an Undirected Graph)
    // --plus
    // --bfs
    public int countComponents(int n, int[][] edges) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] edge : edges) {
            map.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            map.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        int res = 0;
        Queue<Integer> queue = new LinkedList<>();
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                visited[i] = true;
                queue.offer(i);
                while (!queue.isEmpty()) {
                    int node = queue.poll();
                    for (int neighbor : map.getOrDefault(node, new ArrayList<>())) {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            queue.offer(neighbor);
                        }
                    }
                }
                ++res;
            }
        }
        return res;

    }

    public class Union323 {
        private int[] rank;
        private int[] parent;
        private int count;

        public Union323(int n) {
            rank = new int[n];
            Arrays.fill(rank, 1);
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            count = n;
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
            --count;
        }

        public int getCount() {
            return count;
        }

    }

    // 323. 无向图中连通分量的数目 (Number of Connected Components in an Undirected Graph)
    // --plus
    // 并查集
    public int countComponents2(int n, int[][] edges) {
        Union323 union = new Union323(n);
        for (int[] edge : edges) {
            union.union(edge[0], edge[1]);
        }
        return union.getCount();

    }

    // 660. 移除 9 (Remove 9) --plus
    public int newInteger(int n) {
        // 第n个9进制数字
        return Integer.parseInt(Integer.toString(n, 9));
    }

    // 1826. 有缺陷的传感器 (Faulty Sensor) --plus
    public int badSensor(int[] sensor1, int[] sensor2) {
        int n = sensor1.length;
        int i = 0;
        while (i < n) {
            if (sensor1[i] != sensor2[i]) {
                break;
            }
            ++i;
        }
        if (i >= n - 1) {
            return -1;
        }
        int j = i + 1;
        while (j < n) {
            if (sensor1[j - 1] != sensor2[j]) {
                break;
            }
            ++j;
        }
        if (j < n) {
            return 2;
        }
        j = i + 1;
        while (j < n) {
            if (sensor1[j] != sensor2[j - 1]) {
                break;
            }
            ++j;
        }
        if (j < n) {
            return 1;
        }
        return -1;

    }

    // 1427. 字符串的左右移 (Perform String Shifts)
    public String stringShift(String s, int[][] shift) {
        int n = s.length();
        int perform = 0;
        for (int[] shif : shift) {
            if (shif[0] == 0) {
                perform = (perform - shif[1]) % n;
            } else {
                perform = (perform + shif[1]) % n;
            }
        }
        perform = (perform % n + n) % n;
        return s.substring(n - perform) + s.substring(0, n - perform);

    }

    // 1271. 十六进制魔术数字 (Hexspeak)
    public String toHexspeak(String num) {
        Map<Long, Character> map = new HashMap<>();
        map.put(0l, 'O');
        map.put(1l, 'I');
        map.put(10l, 'A');
        map.put(11l, 'B');
        map.put(12l, 'C');
        map.put(13l, 'D');
        map.put(14l, 'E');
        map.put(15l, 'F');
        long n = Long.parseLong(num);
        StringBuilder builder = new StringBuilder();
        while (n != 0l) {
            long mod = n % 16;
            if (!map.containsKey(mod)) {
                return "ERROR";
            }
            builder.append(map.get(mod));
            n /= 16;
        }
        return builder.reverse().toString();

    }

    // 2312. 卖木头块 (Selling Pieces of Wood) --记忆化搜索
    private long[][] memo2312;

    public long sellingWood(int m, int n, int[][] prices) {
        memo2312 = new long[m + 1][n + 1];
        for (int i = 0; i < m + 1; ++i) {
            Arrays.fill(memo2312[i], -1l);
        }
        int[][] values = new int[m + 1][n + 1];
        for (int[] price : prices) {
            values[price[0]][price[1]] = price[2];
        }
        return dfs2312(m, n, values);

    }

    private long dfs2312(int m, int n, int[][] values) {
        if (memo2312[m][n] != -1) {
            return memo2312[m][n];
        }
        long max = values[m][n];
        for (int i = 1; i < m; ++i) {
            max = Math.max(max, dfs2312(i, n, values) + dfs2312(m - i, n, values));
        }
        for (int i = 1; i < n; ++i) {
            max = Math.max(max, dfs2312(m, i, values) + dfs2312(m, n - i, values));
        }
        return memo2312[m][n] = max;
    }

    // 1575. 统计所有可行路径 (Count All Possible Routes) --记忆化搜索
    private int[][] memo1575;
    private final int MOD1575 = (int) (1e9 + 7);

    public int countRoutes(int[] locations, int start, int finish, int fuel) {
        int n = locations.length;
        memo1575 = new int[n][fuel + 1];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo1575[i], -1);
        }
        return dfs1575(locations, start, finish, fuel);
    }

    private int dfs1575(int[] locations, int start, int finish, int fuel) {
        if (memo1575[start][fuel] != -1) {
            return memo1575[start][fuel];
        }
        memo1575[start][fuel] = 0;
        if (Math.abs(locations[start] - locations[finish]) > fuel) {
            return 0;
        }
        for (int i = 0; i < locations.length; ++i) {
            int need = Math.abs(locations[i] - locations[start]);
            if (i != start && need <= fuel) {
                memo1575[start][fuel] += dfs1575(locations, i, finish, fuel - need);
                memo1575[start][fuel] %= MOD1575;
            }
        }
        if (start == finish) {
            memo1575[start][fuel] += 1;
            memo1575[start][fuel] %= MOD1575;
        }
        return memo1575[start][fuel];
    }

    // 294. 翻转游戏 II (Flip Game II) --记忆化搜索
    private Map<String, Boolean> memo294;

    public boolean canWin(String currentState) {
        memo294 = new HashMap<>();
        return dfs294(currentState);

    }

    private boolean dfs294(String currentState) {
        if (memo294.containsKey(currentState)) {
            return memo294.get(currentState);
        }
        for (int i = 1; i < currentState.length(); ++i) {
            if (currentState.charAt(i - 1) == '+' && currentState.charAt(i) == '+') {
                String next = currentState.substring(0, i - 1) + "--" + currentState.substring(i + 1);
                if (!dfs294(next)) {
                    memo294.put(next, false);
                    return true;
                }
                memo294.put(next, true);
            }
        }
        return false;
    }

    // 672. 灯泡开关 Ⅱ (Bulb Switcher II)
    public int flipLights(int n, int presses) {
        if (presses == 0) {
            return 1;
        }
        if (n == 1) {
            return 2;
        }
        if (n == 2) {
            return presses == 1 ? 3 : 4;
        }
        return presses == 1 ? 4 : (presses == 2 ? 7 : 8);
    }

    // 361. 轰炸敌人 (Bomb Enemy) --plus dp
    public int maxKilledEnemies(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] dp = new int[m][n];
        int pre = 0;
        for (int i = 0; i < m; ++i) {
            pre = 0;
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 'W') {
                    pre = 0;
                } else if (grid[i][j] == 'E') {
                    ++pre;
                }
                dp[i][j] += pre;
            }
            pre = 0;
            for (int j = n - 1; j >= 0; --j) {
                if (grid[i][j] == 'W') {
                    pre = 0;
                } else if (grid[i][j] == 'E') {
                    ++pre;
                }
                dp[i][j] += pre;
            }
        }

        for (int j = 0; j < n; ++j) {
            pre = 0;
            for (int i = 0; i < m; ++i) {
                if (grid[i][j] == 'W') {
                    pre = 0;
                } else if (grid[i][j] == 'E') {
                    ++pre;
                }
                dp[i][j] += pre;
            }
            pre = 0;
            for (int i = m - 1; i >= 0; --i) {
                if (grid[i][j] == 'W') {
                    pre = 0;
                } else if (grid[i][j] == 'E') {
                    ++pre;
                }
                dp[i][j] += pre;
            }
        }
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == '0') {
                    res = Math.max(res, dp[i][j]);
                }
            }
        }
        return res;

    }

    // 360. 有序转化数组 (Sort Transformed Array) --plus
    public int[] sortTransformedArray(int[] nums, int a, int b, int c) {
        return a == 0 ? transformLinear(nums, b, c) : transformQuadratic(nums, a, b, c);
    }

    private int[] transformQuadratic(int[] nums, int a, int b, int c) {
        int n = nums.length;
        int[] res = new int[n];

        double symmetryAxis = -b / (2.0 * a);
        int index = a > 0 ? n - 1 : 0;
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            if (Math.abs(nums[left] - symmetryAxis) > Math.abs(nums[right] - symmetryAxis)) {
                res[index] = a * nums[left] * nums[left] + b * nums[left] + c;
                ++left;
            } else {
                res[index] = a * nums[right] * nums[right] + b * nums[right] + c;
                --right;
            }
            index += a > 0 ? -1 : 1;
        }
        return res;

    }

    private int[] transformLinear(int[] nums, int b, int c) {
        int n = nums.length;
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            res[b >= 0 ? i : n - i - 1] = b * nums[i] + c;
        }
        return res;
    }

    // 604. 迭代压缩字符串 (Design Compressed String Iterator) --plus
    class StringIterator {
        private int curLetterIndex;
        private int curCount;
        private int nextLetterIndex;
        private int n;
        private String compressedString;

        public StringIterator(String compressedString) {
            this.compressedString = compressedString;
            this.n = compressedString.length();
            getNext();
        }

        public char next() {
            if (!hasNext()) {
                return ' ';
            }
            if (curCount == 0) {
                getNext();
            }
            --curCount;
            return compressedString.charAt(curLetterIndex);
        }

        public boolean hasNext() {
            return curCount > 0 || nextLetterIndex < n;

        }

        private void getNext() {
            curLetterIndex = nextLetterIndex;
            ++nextLetterIndex;
            curCount = 0;
            while (nextLetterIndex < n && Character.isDigit(compressedString.charAt(nextLetterIndex))) {
                curCount = curCount * 10 + compressedString.charAt(nextLetterIndex) - '0';
                ++nextLetterIndex;
            }
        }
    }

    // 286. 墙与门 (Walls and Gates) --plus bfs
    public void wallsAndGates(int[][] rooms) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = rooms.length;
        int n = rooms[0].length;
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (rooms[i][j] == 0) {
                    queue.offer((i << 8) | j);
                }
            }
        }
        int distance = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            ++distance;
            for (int i = 0; i < size; ++i) {
                int cur = queue.poll();
                for (int[] direction : directions) {
                    int nx = (cur >> 8) + direction[0];
                    int ny = (cur & ((1 << 8) - 1)) + direction[1];
                    if (nx >= 0 && nx < m && ny >= 0 && ny < n && distance < rooms[nx][ny]) {
                        rooms[nx][ny] = distance;
                        queue.offer((nx << 8) | ny);
                    }
                }
            }
        }

    }

    // 1936. 新增的最少台阶数 (Add Minimum Number of Rungs)
    public int addRungs(int[] rungs, int dist) {
        int res = (rungs[0] - 1) / dist;
        for (int i = 1; i < rungs.length; ++i) {
            res += (rungs[i] - rungs[i - 1] - 1) / dist;
        }
        return res;

    }

    // 1922. 统计好数字的数目 (Count Good Numbers) --快速幂
    private final int MOD1922 = (int) (1e9 + 7);

    public int countGoodNumbers(long n) {
        return (int) ((long) quickMul(5, (n + 1) / 2) * quickMul(4, n / 2) % MOD1922);
    }

    private long quickMul(int x, long k) {
        int res = 1;
        int x_contribute = x;
        while (k > 0) {
            if (k % 2 == 1) {
                res = (int) ((long) res * x_contribute % MOD1922);
            }
            x_contribute = (int) ((long) x_contribute * x_contribute % MOD1922);
            k /= 2;
        }
        return res;
    }

    // 2017. 网格游戏 (Grid Game)
    public long gridGame(int[][] grid) {
        int n = grid[0].length;
        long left0 = 0l;
        for (int g : grid[0]) {
            left0 += g;
        }
        long left1 = 0l;
        long res = Long.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            left0 -= grid[0][i];
            res = Math.min(res, Math.max(left0, left1));
            left1 += grid[1][i];
        }
        return res;

    }

    // 6184. 统计共同度过的日子数
    public int countDaysTogether(String arriveAlice, String leaveAlice, String arriveBob, String leaveBob) {
        int[] prefix = { 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        for (int i = 1; i < 13; ++i) {
            prefix[i] += prefix[i - 1];
        }
        int startAlice = getDays6184(arriveAlice, prefix);
        int endAlice = getDays6184(leaveAlice, prefix);
        int startBob = getDays6184(arriveBob, prefix);
        int endBob = getDays6184(leaveBob, prefix);

        return Math.max(0, Math.min(endAlice, endBob) - Math.max(startAlice, startBob) + 1);

    }

    private int getDays6184(String time, int[] prefix) {
        int res = 0;
        int month = Integer.parseInt(time.substring(0, 2));
        res += prefix[month - 1];
        int days = Integer.parseInt(time.substring(3));
        res += days;
        return res;
    }

    // 6185. 运动员和训练师的最大匹配数
    public int matchPlayersAndTrainers(int[] players, int[] trainers) {
        Arrays.sort(players);
        Arrays.sort(trainers);
        int res = 0;

        int n = players.length;
        int m = trainers.length;
        int i = 0;
        int j = 0;
        while (i < n && j < m) {
            if (players[i] <= trainers[j]) {
                ++res;
                ++i;
            }
            ++j;
        }
        return res;

    }

    // 6186. 按位或最大的最小子数组长度
    public int[] smallestSubarrays(int[] nums) {
        int n = nums.length;
        int[] counts = new int[32];
        int[] res = new int[n];
        int j = n - 1;
        int i = n - 1;
        while (i >= 0) {
            addCounts(nums[i], counts);
            while (j > i && tryMine(nums[j], counts.clone())) {
                mine(nums[j--], counts);
            }
            res[i] = j - i + 1;
            --i;
        }
        return res;

    }

    private void mine(int num, int[] counts) {
        int[] c = new int[counts.length];
        int i = 0;
        while (num != 0) {
            c[i++] = num % 2;
            num /= 2;
        }
        for (int ij = 0; ij < counts.length; ++ij) {
            counts[ij] -= c[ij];
        }

    }

    private boolean tryMine(int num, int[] counts) {
        int[] c = new int[counts.length];
        int i = 0;
        while (num != 0) {
            c[i++] = num % 2;
            num /= 2;
        }
        for (int ij = 0; ij < counts.length; ++ij) {
            if (counts[ij] != 0) {
                if (counts[ij] - c[ij] == 0) {
                    return false;
                }
            }
        }
        return true;
    }

    private void addCounts(int num, int[] counts) {
        int i = 0;
        while (num != 0) {
            counts[i++] += num % 2;
            num /= 2;
        }
    }

    // 6180. 最小偶倍数
    public int smallestEvenMultiple(int n) {
        return n % 2 == 0 ? n : n * 2;

    }

    // 6180. 最小偶倍数
    public int smallestEvenMultiple2(int n) {
        return (n % 2 + 1) * n;

    }

    // 6181. 最长的字母序连续子字符串的长度
    public int longestContinuousSubstring(String s) {
        int res = 1;
        int count = 1;
        for (int i = 1; i < s.length(); ++i) {
            if (s.charAt(i) - s.charAt(i - 1) == 1) {
                ++count;
            } else {
                count = 1;
            }
            res = Math.max(res, count);
        }

        return res;

    }

    public TreeNode reverseOddLevels(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        int level = 0;
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            if (level % 2 == 1) {
                List<TreeNode> list = new ArrayList<>(queue);
                int i = 0;
                int j = list.size() - 1;
                while (i < j) {
                    int temp = list.get(i).val;
                    list.get(i).val = list.get(j).val;
                    list.get(j).val = temp;
                    ++i;
                    --j;
                }
                queue = new LinkedList<>(list);
            }
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }

            ++level;

        }
        return root;

    }

    // 6183. 字符串的前缀分数和
    public int[] sumPrefixScores(String[] words) {
        int n = words.length;
        Trie6183 trie = new Trie6183();
        for (String word : words) {
            trie.insert(word);
        }
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            res[i] = trie.getCount(words[i]);
        }
        return res;

    }

    class Trie6183 {
        private Trie6183[] children;
        private int count;

        Trie6183() {
            this.children = new Trie6183[26];
            this.count = 0;
        }

        public void insert(String s) {
            Trie6183 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie6183();
                }
                node = node.children[index];
                ++node.count;
            }
        }

        public int getCount(String s) {
            int res = 0;
            Trie6183 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    break;
                }
                node = node.children[index];
                res += node.count;
            }
            return res;

        }

    }

    // 576. 出界的路径数 (Out of Boundary Paths) --记忆化搜索
    private final int MOD576 = (int) (1e9 + 7);

    public int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
        int[][][] memo = new int[m][n][maxMove + 1];
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                Arrays.fill(memo[i][j], -1);
            }
        }
        return dfs576(m, n, maxMove, startRow, startColumn, memo, directions);

    }

    private int dfs576(int m, int n, int maxMove, int startRow, int startColumn, int[][][] dp, int[][] directions) {
        if (startRow >= m || startRow < 0 || startColumn >= n || startColumn < 0) {
            return 1;
        }
        if (maxMove == 0) {
            return 0;
        }

        if (dp[startRow][startColumn][maxMove] != -1) {
            return dp[startRow][startColumn][maxMove];
        }
        dp[startRow][startColumn][maxMove] = 0;
        for (int[] direction : directions) {
            dp[startRow][startColumn][maxMove] = (dp[startRow][startColumn][maxMove]
                    + dfs576(m, n, maxMove - 1, startRow + direction[0], startColumn + direction[1], dp, directions))
                    % MOD576;
        }
        return dp[startRow][startColumn][maxMove] % MOD576;
    }

    // 334. 递增的三元子序列 (Increasing Triplet Subsequence) --dp
    public boolean increasingTriplet(int[] nums) {
        int n = nums.length;
        int[] leftMax = new int[n];
        leftMax[0] = nums[0];
        for (int i = 1; i < n; ++i) {
            leftMax[i] = Math.min(leftMax[i - 1], nums[i]);
        }
        int[] rightMax = new int[n];
        rightMax[n - 1] = nums[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            rightMax[i] = Math.max(rightMax[i + 1], nums[i]);
        }
        for (int i = 1; i < n - 1; ++i) {
            if (leftMax[i] < nums[i] && nums[i] < rightMax[i]) {
                return true;
            }
        }
        return false;
    }

    // 334. 递增的三元子序列 (Increasing Triplet Subsequence) --贪心
    public boolean increasingTriplet2(int[] nums) {
        int first = nums[0];
        int second = Integer.MAX_VALUE;
        for (int num : nums) {
            if (num > second) {
                return true;
            }
            if (num > first) {
                second = num;
            } else {
                first = num;
            }
        }
        return false;

    }

    // 475. 供暖器 (Heaters)
    public int findRadius(int[] houses, int[] heaters) {
        Arrays.sort(houses);
        Arrays.sort(heaters);

        List<Integer> heaterList = Arrays.stream(heaters).boxed().collect(Collectors.toList());
        heaterList.add(0, Integer.MIN_VALUE);
        heaterList.add(Integer.MAX_VALUE);

        int cur = 0;
        long res = 0;

        for (int i = 0; i < houses.length; ++i) {
            while (cur < heaterList.size()) {
                if (heaterList.get(cur) >= houses[i]) {
                    break;
                }
                ++cur;
            }
            res = Math.max(res,
                    Math.min((long) heaterList.get(cur) - houses[i], (long) houses[i] - heaterList.get(cur - 1)));
        }
        return (int) res;

    }

    // 698. 划分为k个相等的子集 (Partition to K Equal Sum Subsets) --状态压缩 + 记忆化搜索
    private boolean[] memo698;

    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = Arrays.stream(nums).sum();
        if (sum % k != 0) {
            return false;
        }
        Arrays.sort(nums);
        int per = sum / k;
        int n = nums.length;
        if (nums[n - 1] > per) {
            return false;
        }
        this.memo698 = new boolean[1 << n];
        Arrays.fill(memo698, true);

        return dfs698((1 << n) - 1, 0, per, nums);

    }

    private boolean dfs698(int mask, int curSum, int per, int[] nums) {
        if (mask == 0) {
            return true;
        }
        if (!memo698[mask]) {
            return memo698[mask];
        }
        memo698[mask] = false;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] + curSum > per) {
                break;
            }
            if (((mask >> i) & 1) != 0) {
                if (dfs698(mask ^ (1 << i), (nums[i] + curSum) % per, per, nums)) {
                    return true;
                }
            }
        }
        return false;
    }

    // 472. 连接词 (Concatenated Words) --记忆化搜索 + 字典树
    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        Trie472 trie = new Trie472();
        Arrays.sort(words, new Comparator<String>() {

            @Override
            public int compare(String o1, String o2) {
                return o1.length() - o2.length();
            }

        });

        List<String> res = new ArrayList<>();
        for (String word : words) {
            boolean[] visited = new boolean[word.length()];
            if (dfs472(word, trie, visited, 0)) {
                res.add(word);
            } else {
                trie.insert(word);
            }
        }
        return res;

    }

    private boolean dfs472(String word, Trie472 trie, boolean[] visited, int start) {
        if (start == word.length()) {
            return true;
        }
        if (visited[start]) {
            return false;
        }
        Trie472 node = trie;
        visited[start] = true;
        for (int i = start; i < word.length(); ++i) {
            int index = word.charAt(i) - 'a';
            if (node.children[index] == null) {
                return false;
            }
            node = node.children[index];
            if (node.isEnd) {
                if (dfs472(word, trie, visited, i + 1)) {
                    return true;
                }
            }
        }
        return false;
    }

    public class Trie472 {
        private Trie472[] children;
        private boolean isEnd;

        public Trie472() {
            this.children = new Trie472[26];
            this.isEnd = false;
        }

        public void insert(String s) {
            Trie472 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie472();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }

    }

    // 1943. 描述绘画结果 (Describe the Painting) --前缀和 + 差分数组
    public List<List<Long>> splitPainting(int[][] segments) {
        TreeSet<Integer> set = new TreeSet<>();
        int max = 0;
        for (int[] segment : segments) {
            max = Math.max(max, segment[1]);
            set.add(segment[0]);
            set.add(segment[1]);
        }
        long[] diff = new long[max + 1];
        for (int[] segment : segments) {
            diff[segment[0]] += segment[2];
            diff[segment[1]] -= segment[2];
        }

        for (int i = 1; i < diff.length; ++i) {
            diff[i] += diff[i - 1];
        }
        List<List<Long>> res = new ArrayList<>();

        while (set.size() > 1) {
            int left = set.pollFirst();
            int right = set.first();
            if (diff[left] == 0) {
                continue;
            }
            res.add(List.of((long) left, (long) right, diff[left]));
        }
        return res;

    }

    // 2080. 区间内查询数字的频率 (Range Frequency Queries) --二分 + 哈希表
    class RangeFreqQuery {
        private Map<Integer, List<Integer>> map;

        public RangeFreqQuery(int[] arr) {
            map = new HashMap<>();
            for (int i = 0; i < arr.length; ++i) {
                map.computeIfAbsent(arr[i], k -> new ArrayList<>()).add(i);
            }
        }

        public int query(int left, int right, int value) {
            List<Integer> pos = map.getOrDefault(value, new ArrayList<>());
            if (pos.isEmpty()) {
                return 0;
            }
            if (pos.get(0) != Integer.MIN_VALUE) {
                pos.add(0, Integer.MIN_VALUE);
                pos.add(Integer.MAX_VALUE);
            }
            int l = binarySearchLowerBound(left, pos);
            int r = binarySearchUpperBound(right, pos);
            return r - l + 1;
        }

        // pos[i]中的大于等于target的最小索引i
        private int binarySearchLowerBound(int target, List<Integer> pos) {
            int left = 0;
            int right = pos.size() - 1;
            int res = -1;
            while (left <= right) {
                int mid = left + ((right - left) >>> 1);
                if (pos.get(mid) >= target) {
                    res = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            return res;
        }

        // pos[i]中的小于等于target的最大索引i
        private int binarySearchUpperBound(int target, List<Integer> pos) {
            int left = 0;
            int right = pos.size() - 1;
            int res = -1;
            while (left <= right) {
                int mid = left + ((right - left) >>> 1);
                if (pos.get(mid) <= target) {
                    res = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return res;
        }
    }

    // 1744. 你能在你最喜欢的那天吃到你最喜欢的糖果吗？(Can You Eat Your Favorite Candy on Your Favorite
    // Day?)
    public boolean[] canEat(int[] candiesCount, int[][] queries) {
        long[] prefix = new long[candiesCount.length + 1];
        for (int i = 1; i < prefix.length; ++i) {
            prefix[i] = prefix[i - 1] + candiesCount[i - 1];
        }
        int n = queries.length;
        boolean[] res = new boolean[n];
        for (int i = 0; i < n; ++i) {
            int dailyCap = queries[i][2];
            int favoriteDay = queries[i][1];
            int favoriteType = queries[i][0];

            long max = (long) dailyCap * (favoriteDay + 1);
            long min = favoriteDay + 1;

            long actMin = prefix[favoriteType] + 1;
            long actMax = prefix[favoriteType + 1];

            res[i] = min <= actMax && actMin <= max;
        }
        return res;

    }

    // 1016. 子串能表示从 1 到 N 数字的二进制串 (Binary String With Substrings Representing 1 To
    // N)
    public boolean queryString(String s, int n) {
        for (int i = 1; i <= n; ++i) {
            if (!s.contains(Integer.toBinaryString(i))) {
                return false;
            }
        }
        return true;

    }

    // 1911. 最大子序列交替和 (Maximum Alternating Subsequence Sum) --dp
    public long maxAlternatingSum(int[] nums) {
        int n = nums.length;
        long[] even = new long[n];
        even[0] = nums[0];
        long[] odd = new long[n];

        for (int i = 1; i < n; ++i) {
            even[i] = Math.max(odd[i - 1] + nums[i], even[i - 1]);
            odd[i] = Math.max(even[i - 1] - nums[i], odd[i - 1]);
        }
        return even[n - 1];
    }

    // 1911. 最大子序列交替和 (Maximum Alternating Subsequence Sum) --dp
    public long maxAlternatingSum2(int[] nums) {
        int n = nums.length;
        long even = nums[0];
        long odd = 0;
        for (int i = 1; i < n; ++i) {
            long temp = even;
            even = Math.max(odd + nums[i], even);
            odd = Math.max(temp - nums[i], odd);
        }
        return even;
    }

    // 6188. 按身高排序
    public String[] sortPeople(String[] names, int[] heights) {
        int n = names.length;
        List<Node6188> list = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            list.add(new Node6188(names[i], heights[i]));
        }
        Collections.sort(list, new Comparator<Node6188>() {

            @Override
            public int compare(Node6188 o1, Node6188 o2) {
                return o2.height - o1.height;
            }

        });

        String[] res = new String[n];
        for (int i = 0; i < n; ++i) {
            res[i] = list.get(i).name;
        }
        return res;

    }

    class Node6188 {
        public String name;
        public int height;

        public Node6188(String name, int height) {
            this.name = name;
            this.height = height;
        }

    }

    // 6189. 按位与最大的最长子数组
    public int longestSubarray(int[] nums) {
        int n = nums.length;
        int res = 0;
        int cur = 0;
        int max = Arrays.stream(nums).max().getAsInt();
        for (int i = 0; i < n; ++i) {
            if (nums[i] == max) {
                ++cur;
            } else {
                cur = 0;
            }
            res = Math.max(res, cur);

        }
        return res;

    }

    // 6190. 找到所有好下标
    public List<Integer> goodIndices(int[] nums, int k) {
        int n = nums.length;
        int[] left = new int[n];
        Arrays.fill(left, 1);
        for (int i = 2; i < n; ++i) {
            if (nums[i - 1] <= nums[i - 2]) {
                left[i] = left[i - 1] + 1;
            }
        }

        int[] right = new int[n];
        Arrays.fill(right, 1);
        for (int i = n - 3; i >= 0; --i) {
            if (nums[i + 1] <= nums[i + 2]) {
                right[i] = right[i + 1] + 1;
            }
        }

        List<Integer> res = new ArrayList<>();
        for (int i = k; i < n - k; ++i) {
            if (left[i] >= k && right[i] >= k) {
                res.add(i);
            }
        }
        return res;
    }

    // 6191. 好路径的数目
    public int numberOfGoodPaths(int[] vals, int[][] edges) {
        int n = vals.length;
        Union6191 union = new Union6191(n);
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(edge[1]);
            graph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(edge[0]);
        }
        //当前索引对应的值 所在的联通块内等于当前值的个数
        int[] size = new int[n];
        Arrays.fill(size, 1);

        Integer[] indexes = new Integer[n];
        for (int i = 0; i < n; ++i) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return vals[o1] - vals[o2];
            }

        });
        int res = 0;

        for (int x : indexes) {
            int xVal = vals[x];
            int paX = union.getRoot(x);
            for (int y : graph.getOrDefault(x, new ArrayList<>())) {
                int paY = union.getRoot(y);
                if (paX == paY || vals[paY] > xVal) {
                    continue;
                }
                if (vals[paY] == xVal) {
                    res += size[paX] * size[paY];
                    size[paX] += size[paY];
                }
                union.union(paY, paX);
            }
        }
        return res + n;

    }

    public class Union6191 {
        private int[] parent;

        public Union6191(int n) {
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
            parent[p1] = p2;
        }
    }

    // 988. 从叶结点开始的最小字符串 (Smallest String Starting From Leaf) --bfs
    public String smallestFromLeaf(TreeNode root) {
        String res = "";
        Queue<NodeWrapper988> queue = new LinkedList<>();
        queue.offer(new NodeWrapper988(root, "" + (char) ('a' + root.val)));
        while (!queue.isEmpty()) {
            NodeWrapper988 node = queue.poll();
            if (node.node.left == null && node.node.right == null) {
                if (res.isEmpty() || res.compareTo(node.s) > 0) {
                    res = node.s;
                }
            }

            if (node.node.left != null) {
                queue.offer(new NodeWrapper988(node.node.left, (char) ('a' + node.node.left.val) + node.s));
            }
            if (node.node.right != null) {
                queue.offer(new NodeWrapper988(node.node.right, (char) ('a' + node.node.right.val) + node.s));
            }
        }

        return res;

    }

    public class NodeWrapper988 {
        TreeNode node;
        String s;

        public NodeWrapper988(TreeNode node, String s) {
            this.node = node;
            this.s = s;
        }

    }

    // 988. 从叶结点开始的最小字符串 (Smallest String Starting From Leaf) --dfs
    private String res988;

    public String smallestFromLeaf2(TreeNode root) {
        res988 = "";
        dfs988(root, new StringBuilder());
        return res988;

    }

    private void dfs988(TreeNode node, StringBuilder builder) {
        if (node == null) {
            return;
        }
        builder.append((char) (node.val + 'a'));
        if (node.left == null && node.right == null) {
            builder.reverse();
            String S = builder.toString();
            builder.reverse();
            if (res988.isEmpty() || S.compareTo(res988) < 0) {
                res988 = S;
            }
        }
        dfs988(node.left, builder);
        dfs988(node.right, builder);
        builder.deleteCharAt(builder.length() - 1);
    }

    // LCP 61. 气温变化趋势
    public int temperatureTrend(int[] temperatureA, int[] temperatureB) {
        int n = temperatureA.length;
        int res = 0;
        int cur = 0;
        for (int i = 1; i < n; ++i) {
            if (temperatureA[i] > temperatureA[i - 1] && temperatureB[i] > temperatureB[i - 1]
                    || temperatureA[i] == temperatureA[i - 1] && temperatureB[i] == temperatureB[i - 1]
                    || temperatureA[i] < temperatureA[i - 1] && temperatureB[i] < temperatureB[i - 1]) {
                ++cur;
            } else {
                cur = 0;
            }
            res = Math.max(res, cur);
        }
        return res;

    }

    // LCP 62. 交通枢纽
    public int transportationHub(int[][] path) {
        Set<Integer> set = new HashSet<>();
        int[] inDegrees = new int[1001];
        int[] outDegrees = new int[1001];
        for (int[] p : path) {
            ++inDegrees[p[1]];
            ++outDegrees[p[0]];
            set.add(p[1]);
            set.add(p[0]);
        }
        for (int i = 0; i < 1001; ++i) {
            if (inDegrees[i] == set.size() - 1 && outDegrees[i] == 0) {
                return i;
            }
        }
        return -1;

    }

    // LCP 63. 弹珠游戏
    public int[][] ballGame(int num, String[] plate) {
        int m = plate.length;
        int n = plate[0].length();
        List<int[]> res = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                char c = plate[i].charAt(j);
                if (c != '.') {
                    continue;
                }
                if (i == 0 && j != 0 && j != n - 1) {
                    if (checkLCP63(i, j, plate, 0, num)) {
                        res.add(new int[] { i, j });
                    }
                } else if (i == m - 1 && j != 0 && j != n - 1) {
                    if (checkLCP63(i, j, plate, 2, num)) {
                        res.add(new int[] { i, j });
                    }
                } else if (j == 0 && i != 0 && i != m - 1) {
                    if (checkLCP63(i, j, plate, 3, num)) {
                        res.add(new int[] { i, j });
                    }
                } else if (j == n - 1 && i != 0 && i != m - 1) {
                    if (checkLCP63(i, j, plate, 1, num)) {
                        res.add(new int[] { i, j });
                    }
                }
            }
        }
        return res.toArray(new int[res.size()][]);

    }

    private boolean checkLCP63(int i, int j, String[] plate, int d, int step) {
        int m = plate.length;
        int n = plate[0].length();
        // down / left / up / right
        int[][] directions = { { 1, 0 }, { 0, -1 }, { -1, 0 }, { 0, 1 } };
        while (step-- > 0) {
            i += directions[d][0];
            j += directions[d][1];
            if (i < 0 || i >= m || j < 0 || j >= n) {
                return false;
            }
            char c = plate[i].charAt(j);
            if (c == 'O') {
                return true;
            }

            if (c == 'W') {
                d = (d - 1 + 4) % 4;
            } else if (c == 'E') {
                d = (d + 1) % 4;
            }
        }
        return false;

    }

    // 668. 乘法表中第k小的数 (Kth Smallest Number in Multiplication Table) --二分
    public int findKthNumber(int m, int n, int k) {
        int res = -1;
        int left = 1;
        int right = m * n;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (checkCount668(mid, m, n, k)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private boolean checkCount668(int num, int m, int n, int k) {
        int count = 0;
        int i = 1;
        int j = n;
        while (i <= m && j >= 1) {
            if (i * j <= num) {
                count += j;
                ++i;
            } else {
                --j;
            }
        }
        return count >= k;
    }

    // 题目-01. 化学反应
    public int lastMaterial(int[] material) {
        Queue<Integer> queue = new PriorityQueue<>(new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return o2 - o1;
            }

        });

        for (int m : material) {
            queue.offer(m);
        }
        while (queue.size() >= 2) {
            int max = queue.poll();
            int second = queue.poll();
            if (max == second) {
                continue;
            }
            queue.offer(max - second);
        }

        return queue.isEmpty() ? 0 : queue.poll();
    }

    /**
     * The rand7() API is already defined in the parent class SolBase.
     * public int rand7();
     * 
     * @return a random integer in the range 1 to 7
     */

    interface SolBase {
        public int rand7();
    }

    // 470. 用 Rand7() 实现 Rand10() (Implement Rand10() Using Rand7()) --拒绝采样
    class Solution implements SolBase {
        public int rand10() {
            int row;
            int col;
            int idx;
            do {
                row = rand7();
                col = rand7();
                idx = row + (col - 1) * 7;
            } while (idx > 40);
            return 1 + (idx - 1) % 10;
        }

        @Override
        public int rand7() {
            return 0;
        }
    }

    // 1642. 可以到达的最远建筑 (Furthest Building You Can Reach) -- 优先队列 + 贪心
    public int furthestBuilding(int[] heights, int bricks, int ladders) {
        Queue<Integer> queue = new PriorityQueue<>();
        int sum = 0;
        int n = heights.length;
        for (int i = 1; i < n; ++i) {
            int delta = heights[i] - heights[i - 1];
            if (delta > 0) {
                queue.offer(delta);
                if (queue.size() > ladders) {
                    sum += queue.poll();
                }
                if (sum > bricks) {
                    return i - 1;
                }
            }
        }
        return n - 1;

    }

    // 1033. 移动石子直到连续 (Moving Stones Until Consecutive)
    public int[] numMovesStones(int a, int b, int c) {
        int min = 0;
        int[] arr = { a, b, c };
        Arrays.sort(arr);
        if (arr[2] - arr[1] == 1 && arr[1] - arr[0] == 1) {
            min = 0;
        } else if (arr[2] - arr[1] <= 2 || arr[1] - arr[0] <= 2) {
            min = 1;
        } else {
            min = 2;
        }
        int max = (arr[1] - 1) - arr[0] + arr[2] - (arr[1] + 1);
        return new int[] { min, max };

    }

    // 6212. 删除字符使频率相同
    public boolean equalFrequency(String word) {
        int[] counts = new int[26];
        for (char c : word.toCharArray()) {
            ++counts[c - 'a'];
        }
        Arrays.sort(counts);
        int[] copy1 = counts.clone();
        int i = 0;
        while (i < copy1.length) {
            if (copy1[i] != 0) {
                --copy1[i];
                break;
            }
            ++i;
        }
        int[] copy2 = counts.clone();
        --copy2[25];

        if (check6212(copy1) || check6212(copy2)) {
            return true;
        }
        return false;

    }

    private boolean check6212(int[] counts) {
        int same = -1;
        for (int count : counts) {
            if (count != 0) {
                if (same == -1) {
                    same = count;
                } else if (same != count) {
                    return false;
                }
            }
        }
        return true;
    }

    // 6212. 删除字符使频率相同 --暴力
    public boolean equalFrequency2(String word) {
        int[] count = new int[26];
        for (char c : word.toCharArray()) {
            ++count[c - 'a'];
        }
        for (int i = 0; i < count.length; ++i) {
            if (count[i] != 0) {
                --count[i];
                if (check6212_2(count)) {
                    return true;
                }
                ++count[i];
            }
        }
        return false;


    }


    private boolean check6212_2(int[] count) {
        int same = -1;
        for (int c : count) {
            if (c != 0) {
                if (same == -1) {
                    same = c;
                } else if(same != c){
                    return false;
                }
            }
        }
        return true;
    }

    // 6197. 最长上传前缀
    class LUPrefix {
        private int res;
        private Queue<Integer> queue;

        public LUPrefix(int n) {
            queue = new PriorityQueue<>();
        }

        public void upload(int video) {
            queue.offer(video);
            while (!queue.isEmpty() && queue.peek() == res + 1) {
                ++res;
                queue.poll();
            }
        }

        public int longest() {
            return res;

        }
    }
    
    // 6213. 所有数对的异或和
    public int xorAllNums(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int m = nums2.length;
        int xor1 = 0;
        int xor2 = 0;
        for (int num : nums1) {
            xor1 ^= num;
        }
        for (int num : nums2) {
            xor2 ^= num;
        }

        if (m % 2 == 0 && n % 2 == 0) {
            return 0;
        }
        if (m % 2 == 0) {
            return xor2;
        }
        if (n % 2 == 0) {
            return xor1;
        }
        return xor1 ^ xor2;

    }
    
    // 6213. 所有数对的异或和
    public int xorAllNums2(int[] nums1, int[] nums2) {
        int res = 0;
        int m = nums1.length;
        int n = nums2.length;
        if (m % 2 == 1) {
            for (int num : nums2) {
                res ^= num;
            }
        }
        if (n % 2 == 1) {
            for (int num : nums1) {
                res ^= num;
            }
        }
        return res;

    }

    // 6198. 满足不等式的数对数目
    // public long numberOfPairs(int[] nums1, int[] nums2, int diff) {
    //     int n = nums1.length;
    //     int[] d = new int[n];
    //     for (int i = 0; i < n; ++i) {
    //         d[i] = nums1[i] - nums2[i];
    //     }
    //     Queue<Integer> queue = new PriorityQueue<>();
    //     queue.offer(d[0]);
    //     long res = 0l;
    //     for (int j = 1; j < n; ++j) {
    //         int dj = d[j] + diff;
    //         res += binarySearch6198(new ArrayList<>(queue), dj);
    //         queue.offer(d[j]);
    //     }
    //     return res;

    // }

    // // 有序数组中 小于等于target的值有多少个
    // private int binarySearch6198(ArrayList<Integer> list, int target) {
    //     if (target >= list.get(list.size() - 1)) {
    //         return list.size();
    //     }
    //     if (target < list.get(0)) {
    //         return 0;
    //     }

    //     int res = 0;
    //     int left = 0;
    //     int right = list.size() - 1;
    //     while (left <= right) {
    //         int mid = left + ((right - left) >> 1);
    //         if (list.get(mid) <= target) {
    //             res = mid + 1;
    //             left = mid + 1;
    //         } else {
    //             right = mid - 1;
    //         }
    //     }
    //     return res;

    // }

    // 6192. 公因子的数目
    public int commonFactors(int a, int b) {
        int gcd = gcd6192(a, b);
        int res = 0;
        int i = 1;
        while (i * i <= gcd) {
            if (gcd % i == 0) {
                ++res;
                if (i * i < gcd) {
                    ++res;
                }
            }
            ++i;
        }
        return res;

    }

    private int gcd6192(int a, int b) {
        return b == 0 ? a : gcd6192(b, a % b);
    }

    // 6193. 沙漏的最大总和
    public int maxSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        for (int i = 1; i < m - 1; ++i) {
            for (int j = 1; j < n - 1; ++j) {
                int cur = 0;
                cur += grid[i][j];
                cur += grid[i - 1][j];
                cur += grid[i + 1][j];
                cur += grid[i - 1][j - 1];
                cur += grid[i - 1][j + 1];
                cur += grid[i + 1][j - 1];
                cur += grid[i + 1][j + 1];
                res = Math.max(res, cur);
            }
        }
        return res;

    }

    // 6194. 最小 XOR
    public int minimizeXor(int num1, int num2) {
        int count = Integer.bitCount(num2);
        int copy = num1;
        int countNum1 = 0;
        while (copy != 0) {
            ++countNum1;
            copy >>= 1;
        }
        if (count >= countNum1) {
            return (1 << count) - 1;
        }
        int res = 0;
        while (countNum1 > 0) {
            if (count == 0) {
                res <<= 1;
            } else if (count == countNum1) {
                res = (res << 1) | 1;
                --count;
            } else {
                if (((num1 >> (countNum1 - 1)) & 1) == 1) {
                    res = (res << 1) | 1;
                    --count;
                } else {
                    res <<= 1;
                }
            }
            --countNum1;
        }
        return res;

    }
    
    // 6195. 对字母串可执行的最大删除数
    public int deleteString(String s) {
        int n = s.length();
        Set<Character> set = new HashSet<>();
        for (char c : s.toCharArray()) {
            set.add(c);
        }
        if (set.size() == 1) {
            return s.length();
        }
        int[] memo = new int[n + 1];

        Arrays.fill(memo, -1);
        return dfs6195(s, 0, memo);
    }

    private int dfs6195(String s, int start, int[] memo) {
        if (start == s.length() - 1) {
            return 1;
        }
        if (start == s.length()) {
            return 0;
        }
        if (memo[start] != -1) {
            return memo[start];
        }
        int res = 1;
        int span = 1;
        while (start + span * 2 <= s.length()) {
            if (s.substring(start, start + span).equals(s.substring(start + span, start
                    + span * 2))) {
                res = Math.max(res, 1 + dfs6195(s, start + span, memo));
            }
            ++span;
        }

        return memo[start] = res;
    }

    // 1785. 构成特定和需要添加的最少元素 (Minimum Elements to Add to Form a Given Sum)
    public int minElements(int[] nums, int limit, int goal) {
        long sum = 0l;
        for (int num : nums) {
            sum += num;
        }
        long diff = Math.abs(sum - goal);
        if (diff == 0) {
            return 0;
        }
        return (int) ((diff - 1) / limit + 1);

    }

    // 银联-1. 重构链表
    public ListNode reContruct(ListNode head) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode node = dummy;
        while (node.next != null) {
            if (node.next.val % 2 == 0) {
                node.next = node.next.next;
            } else {
                node = node.next;
            }
        }
        return dummy.next;

    }

    // 银联-2. 勘探补给
    public int[] explorationSupply(int[] station, int[] pos) {
        int n = pos.length;
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            res[i] = check_yinlian_2(station, pos[i]);
        }
        return res;

    }

    // station 为排序数组 ，返回接近target值的station[i]的索引i (若距离相同，则选择坐标更小的那一个)
    private int check_yinlian_2(int[] station, int target) {
        int n = station.length;
        if (station[n - 1] <= target) {
            return n - 1;
        }
        if (target <= station[0]) {
            return 0;
        }
        int res = 0;
        int left = 0;
        int right = station.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (station[mid] <= target) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        if (target - station[res] > station[res + 1] - target) {
            return res + 1;
        }
        return res;
    }

    // 银联-3. 风能发电
    public int storedEnergy(int storeLimit, int[] power, int[][] supply) {
        int res = 0;
        for (int i = 0; i < power.length; ++i) {
            int time = binarySearch_yinlian_3(supply, i);
            int min = supply[time][1];
            int max = supply[time][2];
            if (power[i] > max) {
                res = Math.min(storeLimit, res + power[i] - max);
            } else if (power[i] < min) {
                res = Math.max(0, res - (min - power[i]));
            }
        }
        return res;

    }

    // 返回supply[i]中，小雨等于target的最大的i
    private int binarySearch_yinlian_3(int[][] supply, int target) {
        int res = 0;
        int left = 0;
        int right = supply.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (supply[mid][0] <= target) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 九坤-01. 可以读通讯稿的组数
    public int numberOfPairs(int[] nums) {
        int mod = (int) (1e9 + 7);
        int n = nums.length;
        Map<Long, Integer> counts = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            long diff = getCounts_jiukun01(nums[i]);
            counts.put(diff, counts.getOrDefault(diff, 0) + 1);
        }
        long res = 0l;
        for (int count : counts.values()) {
            res = (res + (long) count * (count - 1) / 2) % mod;
        }
        return (int) res;
    }

    private long getCounts_jiukun01(int num) {
        long rev = 0;
        int copy = num;
        while (num != 0) {
            int mod = num % 10;
            rev = rev * 10 + mod;
            num /= 10;
        }
        return copy - rev;
    }

    // 九坤-02. 池塘计数 --dfs
    public int lakeCount(String[] field) {
        int res = 0;
        int m = field.length;
        int n = field[0].length();
        char[][] chars = new char[m][n];
        for (int i = 0; i < m; ++i) {
            chars[i] = field[i].toCharArray();
        }
        int[][] directions = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 }, { 1, 1 }, { -1, -1 }, { 1, -1 }, { -1, 1 } };
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (chars[i][j] == 'W') {
                    dfs_jiukun_02(i, j, chars, directions);
                    ++res;
                }
            }
        }
        return res;
    }

    private void dfs_jiukun_02(int i, int j, char[][] field, int[][] directions) {
        int m = field.length;
        int n = field[0].length;
        field[i][j] = '.';
        for (int[] direction : directions) {
            int nx = i + direction[0];
            int ny = j + direction[1];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n && field[nx][ny] == 'W') {
                dfs_jiukun_02(nx, ny, field, directions);
            }
        }
    }

    // 九坤-02. 池塘计数 --bfs
    public int lakeCount2(String[] field) {
        int res = 0;
        int m = field.length;
        int n = field[0].length();
        char[][] chars = new char[m][n];
        for (int i = 0; i < m; ++i) {
            chars[i] = field[i].toCharArray();
        }
        int[][] directions = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 }, { 1, 1 }, { -1, -1 }, { 1, -1 }, { -1, 1 } };
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (chars[i][j] == 'W') {
                    bfs_jiukun_02(i, j, chars, directions);
                    ++res;
                }
            }
        }
        return res;

    }

    private void bfs_jiukun_02(int i, int j, char[][] field, int[][] directions) {
        int m = field.length;
        int n = field[0].length;
        Queue<Integer> queue = new LinkedList<>();
        field[i][j] = '.';
        queue.offer((i << 7) | j);
        while (!queue.isEmpty()) {
            int mask = queue.poll();
            int x = mask >> 7;
            int y = mask & ((1 << 7) - 1);
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && field[nx][ny] == 'W') {
                    field[nx][ny] = '.';
                    queue.offer((nx << 7) | ny);
                }
            }
        }
    }

    // 九坤-02. 池塘计数 --并查集
    public int lakeCount3(String[] field) {
        int m = field.length;
        int n = field[0].length();
        Union_Find_JiuKun_02 union = new Union_Find_JiuKun_02(m * n);
        int count = 0;

        int[][] directions = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 }, { 1, 1 }, { -1, -1 }, { 1, -1 }, { -1, 1 } };
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (field[i].charAt(j) == 'W') {
                    ++count;
                    for (int[] direction : directions) {
                        int nx = i + direction[0];
                        int ny = j + direction[1];
                        if (nx >= 0 && nx < m && ny >= 0 && ny < n && field[nx].charAt(ny) == 'W') {
                            if (union.union(i * n + j, nx * n + ny)) {
                                --count;
                            }
                        }
                    }
                }
            }
        }

        return count;

    }
    
    public class Union_Find_JiuKun_02 {
        private int[] rank;
        private int[] parent;

        public Union_Find_JiuKun_02(int n) {
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

        public boolean union(int p1, int p2) {
            int root1 = getRoot(p1);
            int root2 = getRoot(p2);
            if (root1 == root2) {
                return false;
            }
            if (rank[root1] < rank[root2]) {
                parent[root1] = root2;
            } else {
                parent[root2] = root1;
                if (rank[root1] == rank[root2]) {
                    ++rank[root1];
                }
            }
            return true;
        }

    }
    
    // 1615. 最大网络秩 (Maximal Network Rank)
    public int maximalNetworkRank(int n, int[][] roads) {
        int[] inDegrees = new int[n];
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for (int[] road : roads) {
            ++inDegrees[road[0]];
            ++inDegrees[road[1]];
            map.computeIfAbsent(road[0], k -> new HashSet<>()).add(road[1]);
            map.computeIfAbsent(road[1], k -> new HashSet<>()).add(road[0]);
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int cur = inDegrees[i] + inDegrees[j] - (map.getOrDefault(i, new HashSet<>()).contains(j) ? 1 : 0);
                res = Math.max(res, cur);
            }
        }
        return res;

    }

     // 1615. 最大网络秩 (Maximal Network Rank)
     public int maximalNetworkRank2(int n, int[][] roads) {
         int[] inDegrees = new int[n];
         int[][] connected = new int[n][n];
         for (int[] road : roads) {
             ++inDegrees[road[0]];
             ++inDegrees[road[1]];
             connected[road[0]][road[1]] = 1;
             connected[road[1]][road[0]] = 1;
         }
         int res = 0;
         for (int i = 0; i < n; ++i) {
             for (int j = i + 1; j < n; ++j) {
                 res = Math.max(res, inDegrees[i] + inDegrees[j] - connected[i][j]);
             }
         }
         return res;
     
    }

    // 1754. 构造字典序最大的合并字符串 (Largest Merge Of Two Strings) --双指针
    public String largestMerge(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        int i = 0;
        int j = 0;
        StringBuilder res = new StringBuilder();
        while (i < m && j < n) {
            if (word1.substring(i).compareTo(word2.substring(j)) > 0) {
                res.append(word1.charAt(i++));
            } else {
                res.append(word2.charAt(j++));
            }
        }
        while (i < m) {
            res.append(word1.charAt(i++));
        }
        while (j < n) {
            res.append(word2.charAt(j++));
        }
        return res.toString();

    }

    // 1871. 跳跃游戏 VII (Jump Game VII)
    public boolean canReach(String s, int minJump, int maxJump) {
        int n = s.length();
        boolean[] dp = new boolean[n];
        dp[0] = true;
        int count = 1;
        for (int i = minJump; i < n; ++i) {
            if (s.charAt(i) == '0' && count > 0) {
                dp[i] = true;
            }
            if (i - maxJump >= 0 && dp[i - maxJump]) {
                --count;
            }
            if (dp[i - minJump + 1]) {
                ++count;
            }
        }
        return dp[n - 1];

    }

    // 811. 子域名访问计数 (Subdomain Visit Count)
    public List<String> subdomainVisits(String[] cpdomains) {
        Map<String, Integer> map = new HashMap<>();
        for (String cpdomain : cpdomains) {
            int spaceIndex = cpdomain.indexOf(" ");
            int count = Integer.parseInt(cpdomain.substring(0, spaceIndex));
            int i = cpdomain.length() - 1;
            while (i >= spaceIndex) {
                if (cpdomain.charAt(i) == '.' || i == spaceIndex) {
                    map.put(cpdomain.substring(i + 1), map.getOrDefault(cpdomain.substring(i + 1), 0) + count);
                }
                --i;
            }
        }
        List<String> res = new ArrayList<>();
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            res.add(entry.getValue() + " " + entry.getKey());
        }
        return res;

    }

    // 1898. 可移除字符的最大数目 (Maximum Number of Removable Characters) --二分查找
    public int maximumRemovals(String s, String p, int[] removable) {
        int res = 0;
        int left = 0;
        int right = removable.length;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (isSubsequence1898(check1898(s, removable, mid), p)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;


    }

    private boolean isSubsequence1898(String s, String p) {
        int i = 0;
        int j = 0;
        while (i < s.length() && j < p.length()) {
            if (s.charAt(i) == p.charAt(j)) {
                ++j;
            }
            ++i;
        }
        return j == p.length();
    }

    private String check1898(String s, int[] removable, int count) {
        char[] chars = s.toCharArray();
        int i = 0;
        while (i < count) {
            chars[removable[i]] = ' ';
            ++i;
        }
        StringBuilder res = new StringBuilder();
        i = 0;
        while (i < chars.length) {
            if (!Character.isWhitespace(chars[i])) {
                res.append(chars[i]);
            }
            ++i;
        }

        return res.toString();
    }

    // 1898. 可移除字符的最大数目 (Maximum Number of Removable Characters) --二分查找
    public int maximumRemovals2(String s, String p, int[] removable) {
        int res = 0;
        int left = 0;
        int right = removable.length;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check1898_2(s, p, removable, mid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private boolean check1898_2(String s, String p, int[] removable, int count) {
        int m = s.length();
        int n = p.length();
        boolean[] deleted = new boolean[m];
        for (int i = 0; i < count; ++i) {
            deleted[removable[i]] = true;
        }
        int i = 0;
        int j = 0;
        while (i < m && j < n) {
            if (!deleted[i] && s.charAt(i) == p.charAt(j)) {
                ++j;
            }
            ++i;
        }
        return j == n;
    }

    // 1283. 使结果不超过阈值的最小除数 (Find the Smallest Divisor Given a Threshold)
    public int smallestDivisor(int[] nums, int threshold) {
        int res = 1;
        int left = 1;
        int right = 0;
        for (int num : nums) {
            right = Math.max(num, right);
        }
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check1283(nums, mid) <= threshold) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;

    }

    private int check1283(int[] nums, int divide) {
        int sum = 0;
        for (int num : nums) {
            sum += (num - 1) / divide + 1;
        }
        return sum;
    }

    // 2029. 石子游戏 IX (Stone Game IX)
    // public boolean stoneGameIX(int[] stones) {

    // }

    // 2049. 统计最高分的节点数目 (Count Nodes With the Highest Score)
    // public int countHighestScoreNodes(int[] parents) {

    // }

    // 827. 最大人工岛 (Making A Large Island)
    // public int largestIsland(int[][] grid) {

    // }

    // 1942. 最小未被占据椅子的编号 (The Number of the Smallest Unoccupied Chair)
    // public int smallestChair(int[][] times, int targetFriend) {

    // }

    // 30. 串联所有单词的子串 (Substring with Concatenation of All Words)
    // public List<Integer> findSubstring(String s, String[] words) {

    // }

    // 932. 漂亮数组 (Beautiful Array)
    // public int[] beautifulArray(int n) {

    // }
}
