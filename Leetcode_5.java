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

public class Leetcode_5 {
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
        for (int i = 1; i < max; ++i) {
            if (Integer.bitCount(i) != numSelect) {
                continue;
            }
            int[] copy = arr.clone();
            int mask = i;
            for (int j = 0; j < n; ++j) {
                if ((mask & (1 << j)) != 0) {
                    for (int k = 0; k < copy.length; ++k) {
                        if ((copy[k] & (1 << j)) != 0) {
                            copy[k] ^= 1 << j;
                        }
                    }
                }
            }
            int cur = 0;
            for (int c : copy) {
                if (c == 0) {
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

    // 2400. 恰好移动 k 步到达某一位置的方法数目 (Number of Ways to Reach a Position After Exactly k
    // Steps)
    // public int numberOfWays(int startPos, int endPos, int k) {

    // }

    // 2402. 会议室 III (Meeting Rooms III)
    // public int mostBooked(int n, int[][] meetings) {

    // }

}
