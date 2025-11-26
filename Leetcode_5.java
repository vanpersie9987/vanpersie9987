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
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@SuppressWarnings("unchecked")
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
    public List<String> ambiguousCoordinates2(String s) {
        List<String> res = new ArrayList<>();
        String digits = s.substring(1, s.length() - 1);
        for (int i = 0; i < digits.length(); ++i) {
            String first = digits.substring(0, i);
            String second = digits.substring(i, digits.length());
            for (String sub1 : getSub(first)) {
                if (!sub1.isEmpty()) {
                    for (String sub2 : getSub(second)) {
                        if (!sub2.isEmpty()) {
                            res.add("(" + sub1 + ", " + sub2 + ")");
                        }
                    }
                }
            }
        }
        return res;

    }

    private List<String> getSub(String s) {
        List<String> res = new ArrayList<>();
        if (s.isEmpty()) {
            return res;
        }
        if (s.charAt(0) != '0' || s.length() == 1) {
            res.add(s);
        }
        for (int i = 1; i < s.length(); ++i) {
            String first = s.substring(0, i);
            // 整数部分
            if (first.charAt(0) == '0' && first.length() > 1) {
                break;
            }
            // 小数部分
            String second = s.substring(i, s.length());
            if (second.charAt(second.length() - 1) == '0') {
                break;
            }
            res.add(first + "." + second);
        }
        return res;
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

    // 813. 最大平均值和的分组 (Largest Sum of Averages)
    private int n813;
    private int[] nums813;
    private int k813;
    private double[][] memo813;

    public double largestSumOfAverages2(int[] nums, int k) {
        this.n813 = nums.length;
        this.k813 = k;
        this.nums813 = nums;
        this.memo813 = new double[n813][k];
        return dfs813(0, 0);

    }

    private double dfs813(int i, int j) {
        if (i == n813) {
            return 0D;
        }
        if (j == k813) {
            return -1e5;
        }
        if (memo813[i][j] != 0D) {
            return memo813[i][j];
        }
        double res = 0D;
        double s = 0D;
        for (int x = i; x <= n813 - k813 + j; ++x) {
            s += nums813[x];
            res = Math.max(res, dfs813(x + 1, j + 1) + s / (x - i + 1));
        }
        return memo813[i][j] = res;
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

    // 2054. 两个最好的不重叠活动 (Two Best Non-Overlapping Events)
    private int n2054;
    private int[][] events2054;
    private int[][] memo2054;

    public int maxTwoEvents2(int[][] events) {
        this.n2054 = events.length;
        Arrays.sort(events, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        this.events2054 = events;
        this.memo2054 = new int[n2054][2];
        for (int i = 0; i < n2054; ++i) {
            Arrays.fill(memo2054[i], -1);
        }
        return dfs2054(0, 0);

    }

    private int dfs2054(int i, int j) {
        if (i == n2054 || j == 2) {
            return 0;
        }
        if (memo2054[i][j] != -1) {
            return memo2054[i][j];
        }
        int p = bisect2054(events2054[i][1] + 1);
        return memo2054[i][j] = Math.max(dfs2054(i + 1, j), dfs2054(p, j + 1) + events2054[i][2]);
    }

    private int bisect2054(int target) {
        if (target <= events2054[0][0]) {
            return 0;
        }
        if (target > events2054[n2054 - 1][0]) {
            return n2054;
        }
        int left = 0;
        int right = n2054 - 1;
        int res = n2054;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (events2054[mid][0] >= target) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
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
        int mx = 0;
        long s = 0L;
        for (int m : milestones) {
            mx = Math.max(mx, m);
            s += m;
        }
        if (mx <= s / 2) {
            return s;
        }
        return (s - mx) * 2 + 1;
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

    // 809. 情感丰富的文字 (Expressive Words)
    public int expressiveWords3(String s, String[] words) {
        List<int[]> list = new ArrayList<>();
        int i = 0;
        while (i < s.length()) {
            char c = s.charAt(i);
            int j = i;
            while (j < s.length() && s.charAt(j) == c) {
                ++j;
            }
            list.add(new int[] { c - 'a', j - i });
            i = j;
        }
        int res = 0;
        search: for (String word : words) {
            i = 0;
            int indexI = 0;
            while (indexI < word.length() && i < list.size()) {
                char c1 = word.charAt(indexI);
                char c2 = (char) (list.get(i)[0] + 'a');
                if (c1 != c2) {
                    continue search;
                }
                int indexJ = indexI;
                while (indexJ < word.length() && word.charAt(indexJ) == c1) {
                    ++indexJ;
                    if (indexJ - indexI > list.get(i)[1]) {
                        continue search;
                    }
                }
                if (list.get(i)[1] != indexJ - indexI && list.get(i)[1] < 3) {
                    continue search;
                }
                indexI = indexJ;
                ++i;
            }
            if (indexI == word.length() && i == list.size()) {
                ++res;
            }
        }
        return res;

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

    // 2389. 和有限的最长子序列 (Longest Subsequence With Limited Sum)
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

    // 2389. 和有限的最长子序列 (Longest Subsequence With Limited Sum) --前缀和 + 二分查找
    public int[] answerQueries2(int[] nums, int[] queries) {
        int n = nums.length;
        int m = queries.length;
        Arrays.sort(nums);
        for (int i = 1; i < n; ++i) {
            nums[i] += nums[i - 1];
        }
        for (int i = 0; i < m; ++i) {
            queries[i] = binarySearch2389(nums, queries[i]);
        }
        return queries;
    }

    /**
     * @return 排序数组nums中，小于等于target的元素个数
     */
    private int binarySearch2389(int[] nums, int target) {
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

    // 2389. 和有限的最长子序列 (Longest Subsequence With Limited Sum) --离线询问
    public int[] answerQueries3(int[] nums, int[] queries) {
        int n = nums.length;
        int m = queries.length;
        int[] res = new int[m];
        Arrays.sort(nums);

        Integer[] ids = IntStream.range(0, m).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return Integer.compare(queries[o1], queries[o2]);
            }

        });
        int i = 0;
        int sum = 0;
        for (int id : ids) {
            while (i < n && sum + nums[i] <= queries[id]) {
                sum += nums[i++];
            }
            res[id] = i;
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

    // 2391. 收集垃圾的最少总时间 (Minimum Amount of Time to Collect Garbage)
    public int garbageCollection(String[] garbage, int[] travel) {
        return cal2391(garbage, travel, 'M') + cal2391(garbage, travel, 'G') + cal2391(garbage, travel, 'P');
    }

    private int cal2391(String[] garbage, int[] travel, char t) {
        int n = garbage.length;
        int res = 0;
        int sum = 0;
        for (int i = n - 1; i >= 0; --i) {
            int g = 0;
            for (char c : garbage[i].toCharArray()) {
                g += c == t ? 1 : 0;
            }
            sum += g;
            if (sum > 0 && i > 0) {
                res += travel[i - 1];
            }
        }
        return res + sum;
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

    // 1567. 乘积为正数的最长子数组长度 (Maximum Length of Subarray With Positive Product)
    private int n1567;
    private int[] nums1567;
    private int[][] memo1567;

    public int getMaxLen3(int[] nums) {
        this.n1567 = nums.length;
        this.nums1567 = nums;
        this.memo1567 = new int[n1567][2];
        for (int i = 0; i < n1567; ++i) {
            Arrays.fill(memo1567[i], -1);
        }
        int res = 0;
        for (int i = 0; i < n1567; ++i) {
            if (nums[i] != 0) {
                res = Math.max(res, dfs1567(i - 1, nums[i] < 0 ? 1 : 0) + 1);
            }
        }
        return res;

    }

    private int dfs1567(int i, int j) {
        if (i < 0 || nums1567[i] == 0) {
            return j == 0 ? 0 : -n1567 - 1;
        }
        if (memo1567[i][j] != -1) {
            return memo1567[i][j];
        }
        return memo1567[i][j] = Math.max(j == 0 ? 0 : -n1567 - 1, dfs1567(i - 1, j ^ (nums1567[i] < 0 ? 1 : 0)) + 1);
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
        int[] row = getTopologicalSort(rowConditions, k);
        int[] col = getTopologicalSort(colConditions, k);
        if (row == null || col == null) {
            return new int[0][];
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < k; ++i) {
            map.put(col[i], i);
        }
        int[][] res = new int[k][k];
        for (int i = 0; i < k; ++i) {
            int num = row[i];
            int colIndex = map.get(num);
            res[i][colIndex] = num;
        }
        return res;

    }

    private int[] getTopologicalSort(int[][] conditions, int k) {
        int[] res = new int[k];
        int[] degrees = new int[k + 1];
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] condition : conditions) {
            graph.computeIfAbsent(condition[0], o -> new ArrayList<>()).add(condition[1]);
            ++degrees[condition[1]];
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 1; i <= k; ++i) {
            if (degrees[i] == 0) {
                queue.offer(i);
            }
        }
        int count = 0;
        while (!queue.isEmpty()) {
            int x = queue.poll();
            res[count++] = x;
            for (int y : graph.getOrDefault(x, new ArrayList<>())) {
                if (--degrees[y] == 0) {
                    queue.offer(y);
                }
            }
        }
        if (count != k) {
            return null;
        }
        return res;
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
        List<String> res = new ArrayList<>();
        boolean inBlock = false;
        StringBuilder sb = new StringBuilder();
        int i = 0;
        while (i < source.length) {
            int j = 0;
            while (j < source[i].length()) {
                if (!inBlock) {
                    if (j + 1 < source[i].length() && source[i].charAt(j) == '/' && source[i].charAt(j + 1) == '*') {
                        inBlock = true;
                        j += 2;
                        continue;
                    }
                    if (j + 1 < source[i].length() && source[i].charAt(j) == '/' && source[i].charAt(j + 1) == '/') {
                        break;
                    }
                    sb.append(source[i].charAt(j));
                    ++j;
                } else {
                    if (j + 1 < source[i].length() && source[i].charAt(j) == '*' && source[i].charAt(j + 1) == '/') {
                        inBlock = false;
                        j += 2;
                        continue;
                    }
                    ++j;
                }
            }
            if (!inBlock && !sb.isEmpty()) {
                res.add(sb.toString());
                sb.setLength(0);
            }
            ++i;
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

    // 983. 最低票价 (Minimum Cost For Tickets)
    private boolean[] arr983;
    private int[] costs983;
    private int[] memo983;

    public int mincostTickets2(int[] days, int[] costs) {
        this.arr983 = new boolean[366];
        for (int d : days) {
            arr983[d] = true;
        }
        this.costs983 = costs;
        this.memo983 = new int[366];
        Arrays.fill(memo983, -1);
        return dfs983(1);
    }

    private int dfs983(int i) {
        if (i > 365) {
            return 0;
        }
        if (memo983[i] != -1) {
            return memo983[i];
        }
        return memo983[i] = Math.min(dfs983(i + 1) + (arr983[i] ? costs983[0] : 0),
                Math.min(dfs983(i + 7) + costs983[1], dfs983(i + 30) + costs983[2]));
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
        if (s.contains("@")) {
            s = s.toLowerCase();
            res.append(s.charAt(0));
            res.append("*****");
            res.append(s.substring(s.indexOf('@') - 1));
        } else {
            int cnt = 0;
            for (int i = s.length() - 1; i >= 0; --i) {
                if (Character.isDigit(s.charAt(i)) && cnt++ < 4) {
                    res.append(s.charAt(i));
                }
            }
            res.append("-***-***");
            cnt -= 10;
            if (cnt > 0) {
                res.append("-");
                while (cnt-- > 0) {
                    res.append("*");
                }
                res.append("+");
            }
            res.reverse();
        }
        return res.toString();

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
        int[] dis = new int[26];
        Arrays.fill(dis, -1);
        for (int i = 0; i < s.length(); ++i) {
            int index = s.charAt(i) - 'a';
            if (dis[index] == -1) {
                dis[index] = i;
            } else if (i - dis[index] - 1 != distance[index]) {
                return false;
            }
        }
        return true;
    }

    // 2401. 最长优雅子数组 (Longest Nice Subarray) 时间：O(n) 空间：O(1)
    public int longestNiceSubarray(int[] nums) {
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

    // 2395. 和相等的子数组 (Find Subarrays With Equal Sum)
    public boolean findSubarrays(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length - 1; ++i) {
            int sum = nums[i] + nums[i + 1];
            if (!set.add(sum)) {
                return true;
            }
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
        int[] arr = new int[m];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                arr[i] |= matrix[i][j] << j;
            }
        }
        int res = 0;
        for (int i = 0; i < (1 << n); ++i) {
            if (Integer.bitCount(i) == numSelect) {
                int cur = 0;
                for (int a : arr) {
                    if ((i & a) == a) {
                        ++cur;
                    }
                }
                res = Math.max(res, cur);
            }
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
            if (!map.containsKey(num.charAt(left)) || !map.containsKey(num.charAt(right))
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

    // 2291. 最大股票收益 (Maximum Profit From Trading Stocks) --plus
    private int n2291;
    private int[] present2291;
    private int[] future2291;
    private int budget2291;
    private int[][] memo2291;

    public int maximumProfit2(int[] present, int[] future, int budget) {
        this.n2291 = present.length;
        this.present2291 = present;
        this.future2291 = future;
        this.budget2291 = budget;
        this.memo2291 = new int[n2291][budget + 1];
        return dfs2291(0, 0);

    }

    private int dfs2291(int i, int used) {
        if (i == n2291) {
            return 0;
        }
        if (memo2291[i][used] != 0) {
            return memo2291[i][used];
        }
        // 不选
        int max = dfs2291(i + 1, used);
        // 选
        if (used + present2291[i] <= budget2291 && future2291[i] - present2291[i] > 0) {
            max = Math.max(max, dfs2291(i + 1, used + present2291[i]) + future2291[i] - present2291[i]);
        }
        return memo2291[i][used] = max;
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

    // 2404. 出现最频繁的偶数元素 (Most Frequent Even Element)
    public int mostFrequentEven(int[] nums) {
        int res = -1;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if ((num & 1) == 0) {
                map.merge(num, 1, Integer::sum);
            }
        }
        int freq = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() > freq) {
                res = entry.getKey();
                freq = entry.getValue();
            } else if (entry.getValue() == freq && entry.getKey() < res) {
                res = entry.getKey();
            }
        }
        return res;

    }

    // 2405. 子字符串的最优划分 (Optimal Partition of String)
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

    // 2405. 子字符串的最优划分 (Optimal Partition of String)
    private int n2405;
    private String s2405;

    public int partitionString2(String s) {
        this.n2405 = s.length();
        this.s2405 = s;
        return dfs2405(0);

    }

    private int dfs2405(int i) {
        if (i == n2405) {
            return 0;
        }
        int mask = 0;
        for (int j = i; j < n2405; ++j) {
            if (((mask >> (s2405.charAt(j) - 'a')) & 1) != 0) {
                return dfs2405(j) + 1;
            }
            mask |= 1 << s2405.charAt(j) - 'a';
        }
        return 1;
    }

    // 2406. 将区间分为最少组数 (Divide Intervals Into Minimum Number of Groups)
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

    // 2406. 将区间分为最少组数 (Divide Intervals Into Minimum Number of Groups)
    public int minGroups2(int[][] intervals) {
        TreeMap<Integer, Integer> diff = new TreeMap<>();
        for (int[] i : intervals) {
            diff.merge(i[0], 1, Integer::sum);
            diff.merge(i[1] + 1, -1, Integer::sum);
        }
        int res = 0;
        int pre = 0;
        for (int d : diff.values()) {
            pre += d;
            res = Math.max(res, pre);
        }
        return res;

    }

    // 2406. 将区间分为最少组数 (Divide Intervals Into Minimum Number of Groups)
    public int minGroups3(int[][] intervals) {
        Arrays.sort(intervals, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        int res = 0;
        Queue<Integer> q = new PriorityQueue<>();
        for (int[] i : intervals) {
            if (!q.isEmpty() && q.peek() < i[0]) {
                q.poll();
            } else {
                ++res;
            }
            q.offer(i[1]);
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

    // 2312. 卖木头块 (Selling Pieces of Wood)
    private long[][] memo2312;
    private int[][] price2312;

    public long sellingWood(int m, int n, int[][] prices) {
        this.memo2312 = new long[m + 1][n + 1];
        for (int i = 0; i < m + 1; ++i) {
            Arrays.fill(memo2312[i], -1L);
        }
        this.price2312 = new int[m + 1][n + 1];
        for (int[] p : prices) {
            price2312[p[0]][p[1]] = p[2];
        }
        return dfs2312(m, n);

    }

    private long dfs2312(int i, int j) {
        if (i == 0 || j == 0) {
            return 0L;
        }
        if (memo2312[i][j] != -1L) {
            return memo2312[i][j];
        }

        long max = price2312[i][j];
        for (int x = 1; x < i; ++x) {
            max = Math.max(max, dfs2312(x, j) + dfs2312(i - x, j));
        }
        for (int y = 1; y < j; ++y) {
            max = Math.max(max, dfs2312(i, y) + dfs2312(i, j - y));
        }
        return memo2312[i][j] = max;
    }

    // 1575. 统计所有可行路径 (Count All Possible Routes) --记忆化搜索
    private int[][] memo1575;
    private int n1575;
    private int[] locations1575;
    private int finish1575;

    public int countRoutes(int[] locations, int start, int finish, int fuel) {
        this.n1575 = locations.length;
        memo1575 = new int[n1575][fuel + 1];
        for (int i = 0; i < n1575; ++i) {
            Arrays.fill(memo1575[i], -1);
        }
        this.locations1575 = locations;
        this.finish1575 = finish;
        return dfs1575(start, fuel);
    }

    private int dfs1575(int i, int j) {
        if (Math.abs(locations1575[i] - locations1575[finish1575]) > j) {
            return 0;
        }
        if (memo1575[i][j] != -1) {
            return memo1575[i][j];
        }
        int res = i == finish1575 ? 1 : 0;
        final int MOD = (int) (1e9 + 7);
        for (int k = 0; k < n1575; ++k) {
            if (k == i) {
                continue;
            }
            res += dfs1575(k, j - Math.abs(locations1575[i] - locations1575[k]));
            res %= MOD;
        }
        return memo1575[i][j] = res;
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
    public int countGoodNumbers(long n) {
        int mod = (int) (1e9 + 7);
        long even = (n + 1) / 2;
        long odd = n - even;
        return (int) ((long) pow1922(5, even) * pow1922(4, odd) % mod);

    }

    private int pow1922(int a, long b) {
        if (b == 0L) {
            return 1;
        }
        final int MOD = (int) (1e9 + 7);
        int res = pow1922(a, b / 2);
        res = (int) ((long) res * res % MOD);
        if (b % 2 == 1) {
            res = (int) ((long) res * a % MOD);
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

    // 2409. 统计共同度过的日子数
    public int countDaysTogether(String arriveAlice, String leaveAlice, String arriveBob, String leaveBob) {
        int[] prefix = { 0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        for (int i = 1; i < prefix.length; ++i) {
            prefix[i] += prefix[i - 1];
        }
        int a1 = getCounts2409(arriveAlice, prefix);
        int a2 = getCounts2409(leaveAlice, prefix);
        int b1 = getCounts2409(arriveBob, prefix);
        int b2 = getCounts2409(leaveBob, prefix);
        return Math.max(0, Math.min(a2, b2) - Math.max(a1, b1) + 1);

    }

    private int getCounts2409(String date, int[] prefix) {
        int month = Integer.parseInt(date.substring(0, 2));
        return prefix[month - 1] + Integer.parseInt(date.substring(3));
    }

    // 2410. 运动员和训练师的最大匹配数 (Maximum Matching of Players With Trainers)
    public int matchPlayersAndTrainers(int[] players, int[] trainers) {
        Arrays.sort(players);
        Arrays.sort(trainers);
        int cnt = 0;
        for (int t : trainers) {
            if (cnt == players.length) {
                break;
            }
            if (players[cnt] <= t) {
                ++cnt;
            }
        }
        return cnt;
    }

    // 2411. 按位或最大的最小子数组长度 (Smallest Subarrays With Maximum Bitwise OR)
    public int[] smallestSubarrays(int[] nums) {
        int n = nums.length;
        int[] pos = new int[31];
        Arrays.fill(pos, -1);
        int[] res = new int[n];
        for (int i = n - 1; i >= 0; --i) {
            int k = i;
            for (int j = 0; j < 31; ++j) {
                if ((nums[i] >> j & 1) != 0) {
                    pos[j] = i;
                }
                if (pos[j] != -1) {
                    k = Math.max(k, pos[j]);
                }
            }
            res[i] = k - i + 1;
        }
        return res;

    }

    // 6180. 最小偶倍数
    public int smallestEvenMultiple(int n) {
        return n % 2 == 0 ? n : n * 2;

    }

    // 6180. 最小偶倍数
    public int smallestEvenMultiple2(int n) {
        return (n % 2 + 1) * n;

    }

    // 2414. 最长的字母序连续子字符串的长度 (Length of the Longest Alphabetical Continuous
    // Substring)
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

    // 2415. 反转二叉树的奇数层 (Reverse Odd Levels of Binary Tree)
    public TreeNode reverseOddLevels(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        int level = 0;
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            if ((level & 1) == 1) {
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
            level ^= 1;
        }
        return root;

    }

    // 2416. 字符串的前缀分数和 (Sum of Prefix Scores of Strings)
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
    private int[][][] memo576;
    private int m576;
    private int n576;

    public int findPaths(int m, int n, int maxMove, int startRow, int startColumn) {
        memo576 = new int[m][n][maxMove + 1];
        this.m576 = m;
        this.n576 = n;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                Arrays.fill(memo576[i][j], -1);
            }
        }
        return dfs576(startRow, startColumn, maxMove);

    }

    private int dfs576(int i, int j, int left) {
        if (i < 0 || i >= m576 || j < 0 || j >= n576) {
            return 1;
        }
        if (left == 0) {
            return 0;
        }
        int min = Math.min(Math.min(i + 1, m576 - i), Math.min(j + 1, n576 - j));
        if (min > left) {
            return 0;
        }
        if (memo576[i][j][left] != -1) {
            return memo576[i][j][left];
        }
        int res = 0;
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        final int MOD = (int) (1e9 + 7);
        for (int[] d : dirs) {
            int nx = d[0] + i;
            int ny = d[1] + j;
            res += dfs576(nx, ny, left - 1);
            res %= MOD;
        }
        return memo576[i][j][left] = res;
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
    private int[] memo698;
    private int u698;
    private int[] maskSum698;
    private int per698;

    public boolean canPartitionKSubsets(int[] nums, int k) {
        int sum = Arrays.stream(nums).sum();
        if (sum % k != 0) {
            return false;
        }
        this.per698 = sum / k;
        int n = nums.length;
        Arrays.sort(nums);
        if (nums[n - 1] > per698) {
            return false;
        }
        this.maskSum698 = new int[1 << n];
        for (int i = 1; i < 1 << n; ++i) {
            int index = Integer.numberOfTrailingZeros(i);
            maskSum698[i] = maskSum698[i ^ (1 << index)] + nums[index];
        }
        this.memo698 = new int[1 << n];
        this.u698 = (1 << n) - 1;
        return dfs698(0);
    }

    private boolean dfs698(int i) {
        if (i == u698) {
            return true;
        }
        if (memo698[i] != 0) {
            return memo698[i] > 0;
        }
        int candidate = i ^ u698;
        for (int c = candidate; c > 0; c = (c - 1) & candidate) {
            if (maskSum698[c] == per698 && dfs698(i | c)) {
                memo698[i] = 1;
                return true;
            }
        }
        memo698[i] = -1;
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

    // 1911. 最大子序列交替和 (Maximum Alternating Subsequence Sum)
    private long[][] memo1911;
    private int[] nums1911;
    private int n1911;

    public long maxAlternatingSum3(int[] nums) {
        this.n1911 = nums.length;
        this.nums1911 = nums;
        this.memo1911 = new long[n1911][2];
        return dfs1911(0, 0);
    }

    private long dfs1911(int i, int j) {
        if (i == n1911) {
            return 0;
        }
        if (memo1911[i][j] != 0L) {
            return memo1911[i][j];
        }
        return memo1911[i][j] = Math.max(dfs1911(i + 1, j), dfs1911(i + 1, j ^ 1) + (1 - 2 * j) * nums1911[i]);
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

    // 2420. 找到所有好下标 (Find All Good Indices)
    public List<Integer> goodIndices(int[] nums, int k) {
        int n = nums.length;
        int[] left = new int[n];
        left[1] = 1;
        for (int i = 2; i < n; ++i) {
            if (nums[i - 1] <= nums[i - 2]) {
                left[i] = left[i - 1] + 1;
            } else {
                left[i] = 1;
            }
        }
        int[] right = new int[n];
        right[n - 2] = 1;
        for (int i = n - 3; i >= 0; --i) {
            if (nums[i + 1] <= nums[i + 2]) {
                right[i] = right[i + 1] + 1;
            } else {
                right[i] = 1;
            }
        }
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
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
        // 当前索引对应的值 所在的联通块内等于当前值的个数
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
            if (temperatureA[i] - temperatureA[i - 1] == temperatureB[i] - temperatureB[i - 1]
                    || (temperatureA[i] - temperatureA[i - 1]) * (temperatureB[i] - temperatureB[i - 1]) > 0) {
                ++cur;
                res = Math.max(res, cur);
            } else {
                cur = 0;
            }
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
     * The rand7() API is already defined in the parent class SolBase. public int
     * rand7();
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

    // 2423. 删除字符使频率相同 (Remove Letter To Equalize Frequency)
    public boolean equalFrequency2(String word) {
        int[] count = new int[26];
        for (char c : word.toCharArray()) {
            ++count[c - 'a'];
        }
        for (int i = 0; i < count.length; ++i) {
            if (count[i] != 0) {
                --count[i];
                if (check2423(count)) {
                    return true;
                }
                ++count[i];
            }
        }
        return false;

    }

    private boolean check2423(int[] count) {
        int same = -1;
        for (int c : count) {
            if (c != 0) {
                if (same == -1) {
                    same = c;
                } else if (same != c) {
                    return false;
                }
            }
        }
        return true;
    }

    // 2423. 删除字符使频率相同 (Remove Letter To Equalize Frequency)
    public boolean equalFrequency(String word) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : word.toCharArray()) {
            map.merge(c, 1, Integer::sum);
        }
        List<Integer> list = new ArrayList<>(map.values());
        Collections.sort(list);
        int size = list.size();
        // 只有一种个数 如 aaaaaaa
        // 有两种个数 且较少的个数是1 较大的个数相等 如 abbbbcccc
        // 有两种个数 且较大的个数比较小的个数大1 较小的个数相等 如 bbbcccdddd
        return size == 1 || list.get(0) == 1 && check2423_2(list.subList(1, size))
                || list.get(size - 1) == list.get(size - 2) + 1 && check2423_2(list.subList(0, size - 1));
    }

    private boolean check2423_2(List<Integer> list) {
        int c = list.get(0);
        for (int cnt : list) {
            if (cnt != c) {
                return false;
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

    // 2430. 对字母串可执行的最大删除数 (Maximum Deletions on a String)
    private int n2430;
    private int[] memo2430;
    private int[][] lcp2430;

    public int deleteString(String s) {
        this.n2430 = s.length();
        this.memo2430 = new int[n2430];
        Arrays.fill(memo2430, -1);
        this.lcp2430 = new int[n2430 + 1][n2430 + 1];
        for (int i = n2430 - 1; i >= 0; --i) {
            for (int j = n2430 - 1; j > i; --j) {
                if (s.charAt(i) == s.charAt(j)) {
                    lcp2430[i][j] = lcp2430[i + 1][j + 1] + 1;
                }
            }
        }
        return dfs2430(0);

    }

    private int dfs2430(int i) {
        if (i == n2430) {
            return 0;
        }
        if (memo2430[i] != -1) {
            return memo2430[i];
        }
        int max = 0;
        for (int j = i + 1; j < n2430 && n2430 - j >= j - i; ++j) {
            if (lcp2430[i][j] >= j - i) {
                max = Math.max(max, dfs2430(j));
            }
        }
        return memo2430[i] = max + 1;
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

    // 1754. 构造字典序最大的合并字符串 (Largest Merge Of Two Strings)
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
        res.append(word1.substring(i));
        res.append(word2.substring(j));
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

    // 1901. 寻找峰值 II (Find a Peak Element II) --二分查找
    public int[] findPeakGrid(int[][] mat) {
        int m = mat.length;
        int left = 0;
        int right = m - 2;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            int colIndex = getMaxColIndex(mat[mid]);
            if (mat[mid][colIndex] > mat[mid + 1][colIndex]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return new int[] { left, getMaxColIndex(mat[left]) };

    }

    private int getMaxColIndex(int[] nums) {
        int index = -1;
        int max = -1;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] > max) {
                max = nums[i];
                index = i;
            }
        }
        return index;
    }

    // 927. 三等分 (Three Equal Parts)
    public int[] threeEqualParts(int[] arr) {
        int n = arr.length;
        int count = 0;
        for (int num : arr) {
            count += num;
        }
        if (count % 3 != 0) {
            return new int[] { -1, -1 };
        }
        if (count == 0) {
            return new int[] { 0, n - 1 };
        }
        count /= 3;
        int i = findFirstOne927(arr, 1);
        int j = findFirstOne927(arr, 1 + count);
        int k = findFirstOne927(arr, 1 + count * 2);
        while (k < n && arr[i] == arr[j] && arr[j] == arr[k]) {
            ++i;
            ++j;
            ++k;
        }
        return k == n ? new int[] { i - 1, j } : new int[] { -1, -1 };

    }

    private int findFirstOne927(int[] arr, int count) {
        int i = 0;
        while (count > 0) {
            if (arr[i] == 1) {
                --count;
            }
            ++i;
        }
        return i;
    }

    // 376. 摆动序列 (Wiggle Subsequence) --空间：O(n) 动态规划
    public int wiggleMaxLength(int[] nums) {
        int n = nums.length;
        if (n <= 1) {
            return n;
        }
        int[] up = new int[n];
        int[] down = new int[n];
        up[0] = down[0] = 1;
        for (int i = 1; i < n; ++i) {
            if (nums[i] > nums[i - 1]) {
                down[i] = Math.max(down[i - 1], up[i - 1] + 1);
                up[i] = up[i - 1];
            } else if (nums[i] < nums[i - 1]) {
                up[i] = Math.max(up[i - 1], down[i - 1] + 1);
                down[i] = down[i - 1];
            } else {
                up[i] = up[i - 1];
                down[i] = down[i - 1];
            }
        }
        return Math.max(up[n - 1], down[n - 1]);

    }

    // 376. 摆动序列 (Wiggle Subsequence) --空间：O(1) 动态规划
    public int wiggleMaxLength2(int[] nums) {
        int n = nums.length;
        if (n <= 1) {
            return n;
        }
        int up = 1;
        int down = 1;
        for (int i = 1; i < n; ++i) {
            if (nums[i] > nums[i - 1]) {
                down = Math.max(down, up + 1);
            } else if (nums[i] < nums[i - 1]) {
                up = Math.max(up, down + 1);
            }
        }
        return Math.max(up, down);

    }

    // 2167. 移除所有载有违禁货物车厢所需的最少时间 (Minimum Time to Remove All Cars Containing Illegal
    // Goods) --前缀和、dp
    public int minimumTime(String s) {

        // (i)+(n−j−1)+2⋅Count(i,j) --Count(i,j)表示 [i,j]之间的1的个数
        // (i)+(n−j−1)+2⋅(pre[j]−pre[i−1])
        // (i−2⋅pre[i−1])+(2⋅pre[j]−j)+(n−1)

        int n = s.length();
        int preBest = 0;
        int preSum = 0;
        int res = Integer.MAX_VALUE;
        for (int j = 0; j < n; ++j) {
            preBest = Math.min(preBest, j - 2 * preSum);
            preSum += s.charAt(j) - '0';
            res = Math.min(res, preBest + 2 * preSum - j);
        }

        return Math.min(res + n - 1, n);

    }

    // 2167. 移除所有载有违禁货物车厢所需的最少时间 (Minimum Time to Remove All Cars Containing Illegal
    // Goods)
    private int n2167;
    private int[] memo_suf2167;
    private int[] memo_pre2167;
    private String s2167;

    public int minimumTime2(String s) {
        this.n2167 = s.length();
        this.s2167 = s;
        this.memo_suf2167 = new int[n2167];
        Arrays.fill(memo_suf2167, -1);
        for (int i = n2167 - 1; i >= 0; --i) {
            dfs_suf2167(i);
        }
        this.memo_pre2167 = new int[n2167];
        Arrays.fill(memo_pre2167, -1);
        int res = n2167;
        for (int i = 0; i < n2167; ++i) {
            res = Math.min(res, dfs_pre2167(i) + dfs_suf2167(i + 1));
        }
        return res;

    }

    private int dfs_pre2167(int i) {
        if (i < 0) {
            return 0;
        }
        if (memo_pre2167[i] != -1) {
            return memo_pre2167[i];
        }
        if (s2167.charAt(i) == '0') {
            return memo_pre2167[i] = dfs_pre2167(i - 1);
        }
        return memo_pre2167[i] = Math.min(i + 1, dfs_pre2167(i - 1) + 2);
    }

    private int dfs_suf2167(int i) {
        if (i == n2167) {
            return 0;
        }
        if (memo_suf2167[i] != -1) {
            return memo_suf2167[i];
        }
        if (s2167.charAt(i) == '0') {
            return memo_suf2167[i] = dfs_suf2167(i + 1);
        }
        return memo_suf2167[i] = Math.min(n2167 - i, dfs_suf2167(i + 1) + 2);

    }

    // 1870. 准时到达的列车最小时速 (Minimum Speed to Arrive on Time) --二分查找
    public int minSpeedOnTime(int[] dist, double hour) {
        int left = 1;
        int right = (int) 1e7;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check1870(mid, dist, hour)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private boolean check1870(int s, int[] dist, double hour) {
        int h = 0;
        for (int i = 0; i < dist.length - 1; ++i) {
            h += (dist[i] - 1) / s + 1;
            if (h > hour) {
                return false;
            }
        }
        return h + (double) dist[dist.length - 1] / s <= hour;
    }

    // 1027. 最长等差数列 (Longest Arithmetic Subsequence) --dp
    public int longestArithSeqLength(int[] nums) {
        int res = 0;
        int n = nums.length;
        // dp[i][d] 以索引i为结尾的元素，公差为d时的最长序列
        int[][] dp = new int[n][1001];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                int d = nums[i] - nums[j] + 500;
                dp[i][d] = dp[j][d] + 1;
                res = Math.max(res, dp[i][d]);
            }
        }
        return res + 1;

    }

    // 1498. 满足条件的子序列数目 (Number of Subsequences That Satisfy the Given Sum
    // Condition) --双指针
    public int numSubseq(int[] nums, int target) {
        int n = nums.length;
        int mod = (int) (1e9 + 7);
        Arrays.sort(nums);
        int[] power = new int[n];
        power[0] = 1;
        for (int i = 1; i < n; ++i) {
            power[i] = (power[i - 1] << 1) % mod;
        }
        int res = 0;
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            if (nums[left] + nums[right] > target) {
                --right;
            } else {
                res = (res + power[right - left]) % mod;
                ++left;
            }
        }
        return res;

    }

    // 1300. 转变数组后最接近目标值的数组和 (Sum of Mutated Array Closest to Target) --二分查找 + 前缀和
    public int findBestValue(int[] arr, int target) {
        int n = arr.length;
        Arrays.sort(arr);
        int[] prefix = new int[n + 1];
        for (int i = 1; i < n + 1; ++i) {
            prefix[i] = prefix[i - 1] + arr[i - 1];
        }
        int res = 0;
        int diff = target;
        int right = arr[n - 1];
        for (int value = 1; value <= right; ++value) {
            int count = binarySearch1300(arr, value);
            int sum = prefix[count] + (n - count) * value;
            if (Math.abs(sum - target) < diff) {
                res = value;
                diff = Math.abs(sum - target);
            }
        }
        return res;
    }

    // 排序数组arr中，小于value的元素个数
    private int binarySearch1300(int[] arr, int value) {
        int n = arr.length;
        if (value <= arr[0]) {
            return 0;
        }
        if (value > arr[n - 1]) {
            return n;
        }
        int res = 0;
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (arr[mid] < value) {
                res = mid + 1;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    // 6200. 处理用时最长的那个任务的员工
    public int hardestWorker(int n, int[][] logs) {
        int max = logs[0][1];
        int res = logs[0][0];
        for (int i = 1; i < logs.length; ++i) {
            int diff = logs[i][1] - logs[i - 1][1];
            if (diff > max) {
                max = diff;
                res = logs[i][0];
            } else if (diff == max) {
                res = Math.min(res, logs[i][0]);
            }
        }
        return res;

    }

    // 6201. 找出前缀异或的原始数组
    public int[] findArray(int[] pref) {
        int n = pref.length;
        int[] res = new int[n];
        res[0] = pref[0];
        for (int i = 1; i < n; ++i) {
            res[i] = pref[i - 1] ^ pref[i];
        }
        return res;

    }

    // 2435. 矩阵中和能被 K 整除的路径 (Paths in Matrix Whose Sum Is Divisible by K)
    private int[][] grid2435;
    private int k2435;
    private int[][][] memo2435;

    public int numberOfPaths(int[][] grid, int k) {
        this.grid2435 = grid;
        int m = grid.length;
        int n = grid[0].length;
        this.k2435 = k;
        this.memo2435 = new int[m][n][k];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                Arrays.fill(memo2435[i][j], -1);
            }
        }
        return dfs2435(m - 1, n - 1, 0);

    }

    private int dfs2435(int i, int j, int m) {
        if (i < 0 || j < 0) {
            return 0;
        }
        m += grid2435[i][j];
        m %= k2435;
        if (i == 0 && j == 0) {
            return m == 0 ? 1 : 0;
        }
        if (memo2435[i][j][m] != -1) {
            return memo2435[i][j][m];
        }
        final int MOD = (int) (1e9 + 7);
        return memo2435[i][j][m] = (dfs2435(i - 1, j, m) + dfs2435(i, j - 1, m)) % MOD;
    }

    // 2435. 矩阵中和能被 K 整除的路径 (Paths in Matrix Whose Sum Is Divisible by K)
    public int numberOfPaths2(int[][] grid, int k) {
        final int mod = (int) (1e9 + 7);
        int m = grid.length;
        int n = grid[0].length;
        int[][][] dp = new int[m][n][k];
        dp[0][0][grid[0][0] % k] = 1;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                for (int x = 0; x < k; ++x) {
                    if (i > 0) {
                        dp[i][j][x] = (dp[i][j][x] + dp[i - 1][j][((x - grid[i][j]) % k + k) % k]) % mod;
                    }
                    if (j > 0) {
                        dp[i][j][x] = (dp[i][j][x] + dp[i][j - 1][((x - grid[i][j]) % k + k) % k]) % mod;
                    }
                }
            }
        }
        return dp[m - 1][n - 1][0];

    }

    // 6202. 使用机器人打印字典序最小的字符串
    public String robotWithString(String s) {
        StringBuilder res = new StringBuilder();
        Stack<Character> stack = new Stack<>();
        int[] count = new int[26];
        for (char c : s.toCharArray()) {
            ++count[c - 'a'];
        }
        int min = 0;
        for (char c : s.toCharArray()) {
            --count[c - 'a'];
            while (min < 26 && count[min] == 0) {
                ++min;
            }
            stack.push(c);
            while (!stack.isEmpty() && stack.peek() - 'a' <= min) {
                res.append(stack.pop());
            }
        }
        return res.toString();
    }

    // 827. 最大人工岛 (Making A Large Island) --并查集 + 枚举
    public int largestIsland(int[][] grid) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int n = grid.length;
        int[] size = new int[n * n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    size[i * n + j] = 1;
                }
            }
        }
        UnionFind827 union = new UnionFind827(size);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    for (int[] direction : directions) {
                        int nx = i + direction[0];
                        int ny = j + direction[1];
                        if (nx >= 0 && nx < n && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                            union.union(i * n + j, nx * n + ny);
                        }
                    }
                }
            }
        }

        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    int root = union.getRoot(i * n + j);
                    res = Math.max(res, union.getCount(root));
                } else {
                    int cur = 1;
                    Set<Integer> set = new HashSet<>();
                    for (int[] direction : directions) {
                        int nx = i + direction[0];
                        int ny = j + direction[1];
                        if (nx >= 0 && nx < n && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                            int root = union.getRoot(nx * n + ny);
                            if (set.add(root)) {
                                cur += union.getCount(root);
                            }
                        }
                    }
                    res = Math.max(res, cur);
                }
            }
        }
        return res;

    }

    public class UnionFind827 {
        private int[] parent;
        private int[] rank;
        private int[] size;

        public UnionFind827(int[] size) {
            int n = size.length;
            this.parent = new int[n];
            this.rank = new int[n];
            this.size = size;

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
            if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
                size[root1] += size[root2];
            } else {
                parent[root1] = root2;
                size[root2] += size[root1];
                if (rank[root1] == rank[root2]) {
                    ++rank[root2];
                }
            }

        }

        public int getCount(int p) {
            int root = getRoot(p);
            return size[root];
        }

    }

    // 1568. 使陆地分离的最少天数 (Minimum Number of Days to Disconnect Island) --bfs + 枚举
    public int minDays(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        if (check1568(grid)) {
            return 0;
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    grid[i][j] = 0;
                    if (check1568(grid)) {
                        return 1;
                    }
                    grid[i][j] = 1;
                }
            }
        }
        return 2;

    }

    private boolean check1568(int[][] grid) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        int x = 0;
        int y = 0;
        int countIsland = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    ++countIsland;
                    x = i;
                    y = j;
                }
            }
        }
        if (countIsland == 0) {
            return true;
        }
        boolean[][] visited = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { x, y });
        visited[x][y] = true;
        --countIsland;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1 && !visited[nx][ny]) {
                    visited[nx][ny] = true;
                    queue.offer(new int[] { nx, ny });
                    --countIsland;
                }
            }
        }
        return countIsland > 0;

    }

    // 801. 使序列递增的最小交换次数 (Minimum Swaps To Make Sequences Increasing) --dp
    public int minSwap(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int[][] dp = new int[n][2];
        dp[0][0] = 0;
        dp[0][1] = 1;
        for (int i = 1; i < n; ++i) {
            int a1 = nums1[i - 1];
            int a2 = nums1[i];
            int b1 = nums2[i - 1];
            int b2 = nums2[i];
            if (a1 < a2 && b1 < b2 && a1 < b2 && b1 < a2) {
                dp[i][0] = Math.min(dp[i - 1][0], dp[i - 1][1]);
                dp[i][1] = Math.min(dp[i - 1][0], dp[i - 1][1]) + 1;
            } else if (a1 < a2 && b1 < b2) {
                dp[i][0] = dp[i - 1][0];
                dp[i][1] = dp[i - 1][1] + 1;
            } else if (a1 < b2 && b1 < a2) {
                dp[i][0] = dp[i - 1][1];
                dp[i][1] = dp[i - 1][0] + 1;
            }
        }
        return Math.min(dp[n - 1][0], dp[n - 1][1]);

    }

    // 801. 使序列递增的最小交换次数 (Minimum Swaps To Make Sequences Increasing) --dp
    public int minSwap2(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int notExchange = 0;
        int exchange = 1;
        for (int i = 1; i < n; ++i) {
            int a1 = nums1[i - 1];
            int a2 = nums1[i];
            int b1 = nums2[i - 1];
            int b2 = nums2[i];
            if (a1 < a2 && b1 < b2 && a1 < b2 && b1 < a2) {
                notExchange = Math.min(notExchange, exchange);
                exchange = notExchange + 1;
            } else if (a1 < a2 && b1 < b2) {
                ++exchange;
            } else if (a1 < b2 && b1 < a2) {
                int temp = notExchange;
                notExchange = exchange;
                exchange = temp + 1;
            }
        }
        return Math.min(notExchange, exchange);

    }

    // 801. 使序列递增的最小交换次数 (Minimum Swaps To Make Sequences Increasing)
    private int n801;
    private int[][] memo801;
    private int[] nums1_801;
    private int[] nums2_801;

    public int minSwap3(int[] nums1, int[] nums2) {
        this.n801 = nums1.length;
        this.nums1_801 = nums1;
        this.nums2_801 = nums2;
        this.memo801 = new int[n801][2];
        for (int i = 0; i < n801; ++i) {
            Arrays.fill(memo801[i], -1);
        }
        return dfs801(0, 0);

    }

    private int dfs801(int i, int j) {
        if (i == n801) {
            return 0;
        }
        if (memo801[i][j] != -1) {
            return memo801[i][j];
        }
        if (nums1_801[i] == nums2_801[i]) {
            return dfs801(i + 1, 0);
        }
        if (i == 0) {
            return Math.min(dfs801(i + 1, 0), dfs801(i + 1, 1) + 1);
        }
        int res = Integer.MAX_VALUE;
        if (nums1_801[i] > nums1_801[i - 1] && nums2_801[i] > nums2_801[i - 1]) {
            res = Math.min(res, dfs801(i + 1, 0) + j);
        }
        if (nums1_801[i] > nums2_801[i - 1] && nums2_801[i] > nums1_801[i - 1]) {
            res = Math.min(res, dfs801(i + 1, j ^ 1) + (j ^ 1));
        }
        return memo801[i][j] = res;
    }

    // 1362. 最接近的因数 (Closest Divisors)
    public int[] closestDivisors(int num) {
        int[] res = new int[] { 1, (int) 1e9 };
        division1362(num + 1, res);
        division1362(num + 2, res);
        return res;

    }

    private void division1362(int target, int[] res) {
        for (int i = (int) Math.sqrt(target); i > 0; --i) {
            if (target % i == 0) {
                if (Math.abs(target / i - i) < Math.abs(res[0] - res[1])) {
                    res[0] = target / i;
                    res[1] = i;
                    return;
                }
            }
        }
    }

    // 2334. 元素值大于变化阈值的子数组 (Subarray With Elements Greater Than Varying Threshold)
    // --并查集
    public int validSubarraySize(int[] nums, int threshold) {
        int n = nums.length;
        UnionFind2334 union = new UnionFind2334(n + 1);
        Integer[] ids = IntStream.range(0, n).boxed().toArray(Integer[]::new);
        Arrays.sort(ids, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return nums[o2] - nums[o1];
            }

        });

        for (int id : ids) {
            union.union(id, id + 1);
            int size = union.getSize(id);
            if (nums[id] > threshold / size) {
                return size;
            }

        }
        return -1;

    }

    public class UnionFind2334 {
        private int[] rank;
        private int[] parent;
        private int[] size;

        public UnionFind2334(int n) {
            this.rank = new int[n];
            Arrays.fill(rank, 1);
            this.parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            this.size = new int[n];
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
                size[root2] += size[root1] + 1;
            } else {
                parent[root2] = root1;
                size[root1] += size[root2] + 1;
                if (rank[root1] == rank[root2]) {
                    ++rank[root1];
                }
            }
        }

        public int getSize(int id) {
            int root = getRoot(id);
            return size[root];

        }

    }

    // 932. 漂亮数组 (Beautiful Array)
    public int[] beautifulArray(int n) {
        if (n == 1) {
            return new int[] { 1 };
        }
        List<Integer> list = new ArrayList<>();
        list.add(1);
        while (list.size() < n) {
            for (int i = 0; i < list.size(); ++i) {
                list.set(i, list.get(i) * 2 - 1);
            }
            int size = list.size();
            for (int i = 0; i < size; ++i) {
                list.add(list.get(i) + 1);
            }
        }
        int[] res = new int[n];
        int i = 0;
        for (int e : list) {
            if (e <= n) {
                res[i++] = e;
            }
        }
        return res;

    }

    // 30. 串联所有单词的子串 (Substring with Concatenation of All Words)
    public List<Integer> findSubstring(String s, String[] words) {
        int n = s.length();
        int m = words.length;
        int w = words[0].length();
        List<Integer> res = new ArrayList<>();
        Map<String, Integer> count = new HashMap<>();
        for (String word : words) {
            count.put(word, count.getOrDefault(word, 0) + 1);
        }
        search: for (int i = 0; i + m * w <= n; ++i) {
            Map<String, Integer> curCount = new HashMap<>();
            String sub = s.substring(i, i + m * w);
            for (int j = 0; j + w <= m * w; j += w) {
                String word = sub.substring(j, j + w);
                if (!count.containsKey(word)) {
                    continue search;
                }
                curCount.put(word, curCount.getOrDefault(word, 0) + 1);
            }
            if (count.equals(curCount)) {
                res.add(i);
            }
        }
        return res;

    }

    // 2245. 转角路径的乘积中最多能有几个尾随零 (Maximum Trailing Zeros in a Cornered Path)
    public int maxTrailingZeros(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int count2 = calCount(grid[i][j], 2);
                int count5 = calCount(grid[i][j], 5);
                grid[i][j] = (count2 << 16) | count5;
            }
        }
        // grid[i][j]的左边，因子2的个数
        int[][] left2 = new int[m][n];
        // grid[i][j]的右边，因子2的个数
        int[][] right2 = new int[m][n];
        // grid[i][j]的上边，因子2的个数
        int[][] up2 = new int[m][n];
        // grid[i][j]的下边，因子2的个数
        int[][] down2 = new int[m][n];
        // grid[i][j]的左边，因子5的个数
        int[][] left5 = new int[m][n];
        // grid[i][j]的右边，因子5的个数
        int[][] right5 = new int[m][n];
        // grid[i][j]的上边，因子5的个数
        int[][] up5 = new int[m][n];
        // grid[i][j]的下边，因子5的个数
        int[][] down5 = new int[m][n];

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != 0) {
                    up2[i][j] = up2[i - 1][j] + (grid[i - 1][j] >> 16);
                    up5[i][j] = up5[i - 1][j] + (((1 << 16) - 1) & grid[i - 1][j]);
                }
                if (j != 0) {
                    left2[i][j] = left2[i][j - 1] + (grid[i][j - 1] >> 16);
                    left5[i][j] = left5[i][j - 1] + (((1 << 16) - 1) & grid[i][j - 1]);
                }
            }
        }

        for (int i = m - 1; i >= 0; --i) {
            for (int j = n - 1; j >= 0; --j) {
                if (i != m - 1) {
                    down2[i][j] = down2[i + 1][j] + (grid[i + 1][j] >> 16);
                    down5[i][j] = down5[i + 1][j] + (((1 << 16) - 1) & grid[i + 1][j]);
                }
                if (j != n - 1) {
                    right2[i][j] = right2[i][j + 1] + (grid[i][j + 1] >> 16);
                    right5[i][j] = right5[i][j + 1] + (((1 << 16) - 1) & grid[i][j + 1]);
                }
            }
        }

        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int count1 = Math.min(left2[i][j] + up2[i][j] + (grid[i][j] >> 16),
                        left5[i][j] + up5[i][j] + (((1 << 16) - 1) & grid[i][j]));
                int count2 = Math.min(up2[i][j] + right2[i][j] + (grid[i][j] >> 16),
                        up5[i][j] + right5[i][j] + (((1 << 16) - 1) & grid[i][j]));
                int count3 = Math.min(right2[i][j] + down2[i][j] + (grid[i][j] >> 16),
                        right5[i][j] + down5[i][j] + (((1 << 16) - 1) & grid[i][j]));
                int count4 = Math.min(down2[i][j] + left2[i][j] + (grid[i][j] >> 16),
                        down5[i][j] + left5[i][j] + (((1 << 16) - 1) & grid[i][j]));
                res = Math.max(res, Math.max(Math.max(count1, count2), Math.max(count3, count4)));
            }
        }
        return res;

    }

    private int calCount(int num, int factor) {
        int count = 0;
        while (num % factor == 0) {
            ++count;
            num /= factor;
        }
        return count;
    }

    // 1139. 最大的以 1 为边界的正方形 (Largest 1-Bordered Square)
    public int largest1BorderedSquare(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] preRow = new int[m][n + 1];
        int[][] preCol = new int[n][m + 1];
        for (int i = 0; i < m; ++i) {
            for (int j = 1; j < n + 1; ++j) {
                preRow[i][j] = preRow[i][j - 1] + grid[i][j - 1];
            }
        }
        for (int j = 0; j < n; ++j) {
            for (int i = 1; i < m + 1; ++i) {
                preCol[j][i] = preCol[j][i - 1] + grid[i - 1][j];
            }
        }
        for (int k = Math.min(m, n); k > 0; --k) {
            for (int i = 0; i + k - 1 < m; ++i) {
                for (int j = 0; j + k - 1 < n; ++j) {
                    if (preRow[i][j + k] - preRow[i][j] == k && preRow[i + k - 1][j + k] - preRow[i + k - 1][j] == k
                            && preCol[j][i + k] - preCol[j][i] == k
                            && preCol[j + k - 1][i + k] - preCol[j + k - 1][i] == k) {
                        return k * k;
                    }
                }
            }
        }
        return 0;

    }

    // 940. 不同的子序列 II (Distinct Subsequences II)
    public int distinctSubseqII(String s) {
        final int mod = (int) (1e9 + 7);
        int[] last = new int[26];
        int res = 1;
        for (int i = 0; i < s.length(); ++i) {
            int newCount = res;
            res = ((res % mod + newCount % mod) % mod - last[s.charAt(i) - 'a'] % mod + mod) % mod;
            last[s.charAt(i) - 'a'] = newCount;
        }
        return res - 1;

    }

    // 2437. 有效时间的数目 (Number of Valid Clock Times)
    public int countTime(String time) {
        String h = time.substring(0, 2);
        int count1 = getCounts2437(h, 23);
        String m = time.substring(3);
        int count2 = getCounts2437(m, 59);
        return count1 * count2;

    }

    private int getCounts2437(String h, int limit) {
        int count = 0;
        search: for (int i = 0; i <= limit; ++i) {
            String t = String.format("%02d", i);
            for (int j = 0; j < h.length(); ++j) {
                if (h.charAt(j) == '?') {
                    continue;
                }
                if (h.charAt(j) != t.charAt(j)) {
                    continue search;
                }
            }
            ++count;
        }
        return count;
    }

    // 2437. 有效时间的数目 (Number of Valid Clock Times)
    public int countTime2(String time) {
        String h = time.substring(0, 2);
        String m = time.substring(3);
        int c1 = 0;
        int c2 = 0;
        if (h.charAt(0) == '?' && h.charAt(1) == '?') {
            c1 = 24;
        } else if (h.charAt(0) == '?') {
            if (h.charAt(1) >= '0' && h.charAt(1) <= '3') {
                c1 = 3;
            } else {
                c1 = 2;
            }
        } else if (h.charAt(1) == '?') {
            if (h.charAt(0) >= '0' && h.charAt(0) <= '1') {
                c1 = 10;
            } else {
                c1 = 4;
            }
        } else {
            c1 = 1;
        }

        if (m.charAt(0) == '?' && m.charAt(1) == '?') {
            c2 = 60;
        } else if (m.charAt(0) == '?') {
            c2 = 6;
        } else if (m.charAt(1) == '?') {
            c2 = 10;
        } else {
            c2 = 1;
        }
        return c1 * c2;

    }

    // 2438. 二的幂数组中查询范围内的乘积 (Range Product Queries of Powers)
    public int[] productQueries(int n, int[][] queries) {
        int[] a = new int[Integer.bitCount(n) + 1];
        int id = 1;
        while (n != 0) {
            int lb = Integer.numberOfTrailingZeros(n);
            a[id] = a[id - 1] + lb;
            ++id;
            n &= n - 1;
        }
        int[] pow = new int[a[a.length - 1] + 1];
        pow[0] = 1;
        final int MOD = (int) (1e9 + 7);
        for (int i = 1; i < a[a.length - 1] + 1; ++i) {
            pow[i] = (pow[i - 1] << 1) % MOD;
        }
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            int x = queries[i][0];
            int y = queries[i][1];
            res[i] = pow[a[y + 1] - a[x]];
        }
        return res;
    }

    // 2439. 最小化数组中的最大值 (Minimize Maximum of Array)
    public int minimizeArrayValue(int[] nums) {
        int left = 0;
        int right = 0;
        int res = -1;
        for (int i = 0; i < nums.length; ++i) {
            right = Math.max(right, nums[i]);
            left = Math.min(left, nums[i]);
        }
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (check2439(mid, nums)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return (int) res;

    }

    private boolean check2439(int target, int[] nums) {
        long have = 0l;
        for (int i = nums.length - 1; i >= 0; --i) {
            if (nums[i] > target) {
                have += nums[i] - target;
            } else {
                have -= Math.min(have, target - nums[i]);
            }
        }
        return have <= 0;
    }

    // 2441. 与对应负数同时存在的最大正整数 (Largest Positive Integer That Exists With Its
    // Negative)
    public int findMaxK(int[] nums) {
        Set<Integer> set = Arrays.stream(nums).boxed().collect(Collectors.toSet());
        for (int i = 1000; i >= 1; --i) {
            if (set.contains(i) && set.contains(-i)) {
                return i;
            }
        }
        return -1;

    }

    // 2441. 与对应负数同时存在的最大正整数 (Largest Positive Integer That Exists With Its
    // Negative)
    public int findMaxK2(int[] nums) {
        Set<Integer> set = new HashSet<>();
        int res = -1;
        for (int num : nums) {
            set.add(num);
            if (set.contains(-num)) {
                res = Math.max(res, Math.abs(num));
            }
        }
        return res;

    }

    // 2442. 反转之后不同整数的数目 (Count Number of Distinct Integers After Reverse
    // Operations)
    public int countDistinctIntegers(int[] nums) {
        Set<Integer> set = Arrays.stream(nums).boxed().collect(Collectors.toSet());
        for (int num : nums) {
            set.add(reverse2442(num));
        }
        return set.size();

    }

    private int reverse2442(int num) {
        int res = 0;
        while (num != 0) {
            int mod = num % 10;
            res = res * 10 + mod;
            num /= 10;
        }

        return res;
    }

    // 2443. 反转之后的数字和 (Sum of Number and Its Reverse)
    public boolean sumOfNumberAndReverse(int num) {
        for (int i = 0; i <= num; ++i) {
            if (reverse2443(i) + i == num) {
                return true;
            }

        }
        return false;

    }

    private int reverse2443(int num) {
        int res = 0;
        while (num != 0) {
            int mod = num % 10;
            res = res * 10 + mod;
            num /= 10;
        }

        return res;
    }

    // 2444. 统计定界子数组的数目 (Count Subarrays With Fixed Bounds)
    public long countSubarrays(int[] nums, int minK, int maxK) {
        int n = nums.length;
        int minIndex = -1;
        int maxIndex = -1;
        int d = -1;
        long res = 0l;
        for (int i = 0; i < n; ++i) {
            if (nums[i] > maxK || nums[i] < minK) {
                minIndex = -1;
                maxIndex = -1;
                d = i;
                continue;
            }
            if (nums[i] == minK) {
                minIndex = i;
            }
            if (nums[i] == maxK) {
                maxIndex = i;
            }
            if (minIndex != -1 && maxIndex != -1) {
                res += Math.min(minIndex, maxIndex) - d;
            }
        }
        return res;

    }

    // 497. 非重叠矩形中的随机点 (Random Point in Non-overlapping Rectangles)
    class Solution497 {
        private int[][] rects;
        private int[] prefix;
        private Random random;

        public Solution497(int[][] rects) {
            int n = rects.length;
            this.rects = rects;
            this.prefix = new int[n + 1];
            for (int i = 1; i <= n; ++i) {
                int count = (rects[i - 1][3] - rects[i - 1][1] + 1) * (rects[i - 1][2] - rects[i - 1][0] + 1);
                prefix[i] = prefix[i - 1] + count;
            }
            this.random = new Random();

        }

        public int[] pick() {
            int target = random.nextInt(prefix[prefix.length - 1]) + 1;
            int index = binarySearch497(target) - 1;
            int[] arr = rects[index];
            int num = target - prefix[index] - 1;
            int col = arr[3] - arr[1] + 1;
            int da = num / col;
            int db = num - col * da;
            return new int[] { arr[0] + da, arr[1] + db };

        }

        // 返回单调递增数组prefix中，大于等于target的最小数的索引
        private int binarySearch497(int target) {
            int res = -1;
            int left = 1;
            int right = prefix.length - 1;
            while (left <= right) {
                int mid = left + (right - left >>> 1);
                if (prefix[mid] >= target) {
                    res = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            return res;

        }
    }

    // 1759. 统计同构子字符串的数目 (Count Number of Homogenous Substrings)
    public int countHomogenous(String s) {
        final int mod = (int) (1e9 + 7);
        int res = 0;
        int count = 0;
        char pre = 'A';
        for (char c : s.toCharArray()) {
            if (c == pre) {
                ++count;
            } else {
                count = 1;
                pre = c;
            }
            res = (res + count) % mod;
        }
        return res;

    }

    // 1780. 判断一个数字是否可以表示成三的幂的和 (Check if Number is a Sum of Powers of Three)
    public boolean checkPowersOfThree(int n) {
        while (n != 0) {
            if (n % 3 == 2) {
                return false;
            }
            n /= 3;
        }
        return true;

    }

    // 1737. 满足三条件之一需改变的最少字符数 (Change Minimum Characters to Satisfy One of Three)
    public int minCharacters(String a, String b) {
        int lenA = a.length();
        int lenB = b.length();
        int[] countsA = getCounts1737(a);
        int[] countsB = getCounts1737(b);

        int res = Integer.MAX_VALUE;

        int prefixA = 0;
        int prefixB = 0;

        for (int i = 0; i < 26; ++i) {
            // condition 3
            int leftA = lenA - countsA[i];
            int leftB = lenB - countsB[i];
            res = Math.min(res, leftA + leftB);

            // 不存在严格小于'a'的字符
            if (i == 0) {
                continue;
            }
            // 字符串 a 中，严格小于 ((char) (i + 'a'))的字母的个数
            prefixA += countsA[i - 1];
            // 字符串 b 中，严格小于 ((char) (i + 'a'))的字母的个数
            prefixB += countsB[i - 1];
            res = Math.min(res, lenA - prefixA + prefixB);
            res = Math.min(res, lenB - prefixB + prefixA);
        }
        return res;

    }

    private int[] getCounts1737(String s) {
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        return counts;
    }

    // 1864. 构成交替字符串需要的最小交换次数 (Minimum Number of Swaps to Make the Binary String)
    public int minSwaps(String s) {
        int n = s.length();
        int[] counts = new int[2];
        for (char c : s.toCharArray()) {
            ++counts[c - '0'];
        }
        if (Math.abs(counts[0] - counts[1]) > 1) {
            return -1;
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (((i % 2) ^ (s.charAt(i) - '0')) == 1) {
                ++res;
            }
        }

        if (counts[0] > counts[1]) {
            return res / 2;
        } else if (counts[0] < counts[1]) {
            return (n - res) / 2;
        }
        return Math.min(res / 2, (n - res) / 2);

    }

    // 1856. 子数组最小乘积的最大值 (Maximum Subarray Min-Product) --单调栈 + 前缀和
    public int maxSumMinProduct(int[] nums) {
        int n = nums.length;
        long[] pre = new long[n + 1];
        for (int i = 0; i < n; ++i) {
            pre[i + 1] = pre[i] + nums[i];
        }
        long res = 0L;
        int[] left = new int[n];
        Arrays.fill(left, -1);
        Stack<Integer> st = new Stack<>();
        for (int i = 0; i < n; ++i) {
            while (!st.isEmpty() && nums[st.peek()] >= nums[i]) {
                st.pop();
            }
            if (!st.isEmpty()) {
                left[i] = st.peek();
            }
            st.push(i);
        }

        int[] right = new int[n];
        Arrays.fill(right, n);
        st.clear();
        for (int i = n - 1; i >= 0; --i) {
            while (!st.isEmpty() && nums[st.peek()] >= nums[i]) {
                st.pop();
            }
            if (!st.isEmpty()) {
                right[i] = st.peek();
            }
            st.push(i);
        }
        for (int i = 0; i < n; ++i) {
            res = Math.max(res, nums[i] * (pre[right[i]] - pre[left[i] + 1]));
        }
        final long MOD = (long) (1e9 + 7);
        return (int) (res % MOD);

    }

    // 1235. 规划兼职工作 (Maximum Profit in Job Scheduling)
    public int jobScheduling(int[] startTime, int[] endTime, int[] profit) {
        int n = startTime.length;
        int[][] jobs = new int[n][];
        for (int i = 0; i < n; ++i) {
            jobs[i] = new int[] { startTime[i], endTime[i], profit[i] };
        }
        Arrays.sort(jobs, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }

        });

        // dp[i] : 前 i 份工作可获得的最大利润
        int[] dp = new int[n + 1];
        for (int i = 0; i < n; ++i) {
            int j = binarySearch1235(jobs, i, jobs[i][0]);
            dp[i + 1] = Math.max(dp[i], dp[j + 1] + jobs[i][2]);
        }
        return dp[n];

    }

    private int binarySearch1235(int[][] jobs, int right, int upper) {
        int res = -1;
        int left = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (jobs[mid][1] <= upper) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    // 1235. 规划兼职工作 (Maximum Profit in Job Scheduling)
    private int n1235;
    private int[] memo1235;
    private int[][] arr1235;

    public int jobScheduling2(int[] startTime, int[] endTime, int[] profit) {
        this.n1235 = startTime.length;
        this.memo1235 = new int[n1235];
        this.arr1235 = new int[n1235][3];
        for (int i = 0; i < n1235; ++i) {
            arr1235[i][0] = startTime[i];
            arr1235[i][1] = endTime[i];
            arr1235[i][2] = profit[i];
        }
        Arrays.sort(arr1235, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });
        return dfs1235(0);

    }

    private int dfs1235(int i) {
        if (i == n1235) {
            return 0;
        }
        if (memo1235[i] != 0) {
            return memo1235[i];
        }
        int p = bisect1235(arr1235[i][1]);
        return memo1235[i] = Math.max(dfs1235(i + 1), dfs1235(p) + arr1235[i][2]);
    }

    private int bisect1235(int target) {
        if (target <= arr1235[0][0]) {
            return 0;
        }
        if (target > arr1235[n1235 - 1][0]) {
            return n1235;
        }
        int left = 0;
        int right = n1235 - 1;
        int res = n1235;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (arr1235[mid][0] >= target) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    // 221021天池-01. 统计链表奇数节点
    public int numberEvenListNode(ListNode head) {
        int res = 0;
        while (head != null) {
            if (head.val % 2 == 1) {
                ++res;
            }
            head = head.next;
        }
        return res;

    }

    // 221021天池-02. 光线反射
    public int getLength(String[] grid) {
        int m = grid.length;
        int n = grid[0].length();
        // up , down , left , right
        int[][] directions = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        int d = 1;
        int res = 0;
        int i = 0;
        int j = 0;
        while (i >= 0 && j >= 0 && i < m && j < n) {
            ++res;
            if (grid[i].charAt(j) == 'L') {
                d ^= 2;
            } else if (grid[i].charAt(j) == 'R') {
                d ^= 3;
            }
            i += directions[d][0];
            j += directions[d][1];
        }
        return res;

    }

    // 221021天池-03. 整理书架
    public int[] arrangeBookshelf(int[] order, int limit) {
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        int[] leftCount = new int[1000001];
        for (int c : order) {
            ++leftCount[c];
        }
        int[] stackCount = new int[1000001];
        for (int c : order) {
            if (stackCount[c] == limit) {
                --leftCount[c];
                continue;
            }
            while (c < stack.peek() && leftCount[stack.peek()] > limit) {
                --leftCount[stack.peek()];
                --stackCount[stack.peek()];
                stack.pop();
            }
            stack.push(c);
            stackCount[c] += 1;
        }
        int n = stack.size() - 1;
        int i = 0;
        int[] res = new int[n];
        while (stack.size() > 1) {
            res[n - i - 1] = stack.pop();
            ++i;
        }
        return res;

    }

    // 992. K 个不同整数的子数组 (Subarrays with K Different Integers)
    public int subarraysWithKDistinct(int[] nums, int k) {
        return getSubarraysWithMostKKinds992(nums, k) - getSubarraysWithMostKKinds992(nums, k - 1);

    }

    private int getSubarraysWithMostKKinds992(int[] nums, int k) {
        if (k == 0) {
            return 0;
        }
        int n = nums.length;
        int[] counts = new int[n + 1];
        int kinds = 0;
        int left = 0;
        int right = 0;
        int res = 0;
        while (right < n) {
            if (counts[nums[right]]++ == 0) {
                ++kinds;
            }
            while (kinds > k) {
                if (--counts[nums[left++]] == 0) {
                    --kinds;
                }
            }
            res += right - left + 1;
            ++right;
        }
        return res;
    }

    // 6214. 判断两个事件是否存在冲突
    public boolean haveConflict(String[] event1, String[] event2) {
        return !(event1[1].compareTo(event2[0]) < 0 || event2[1].compareTo(event1[0]) < 0);

    }

    // 6224. 最大公因数等于 K 的子数组数目
    public int subarrayGCD(int[] nums, int k) {
        int n = nums.length;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] % k != 0) {
                continue;
            }
            int gcd = 0;
            for (int j = i; j < n; ++j) {
                gcd = gcd6224(gcd, nums[j]);
                if (gcd == k) {
                    ++res;
                } else if (gcd % k != 0) {
                    break;
                }
            }
        }
        return res;

    }

    private int gcd6224(int a, int b) {
        return b == 0 ? a : gcd6224(b, a % b);
    }

    // 6216. 使数组相等的最小开销
    public long minCost(int[] nums, int[] cost) {
        int n = nums.length;
        long[][] pairs = new long[n][2];
        long sumCost = 0l;
        long cur = 0l;
        for (int i = 0; i < n; ++i) {
            pairs[i][0] = nums[i];
            pairs[i][1] = cost[i];
            sumCost += cost[i];
        }
        Arrays.sort(pairs, new Comparator<long[]>() {

            @Override
            public int compare(long[] o1, long[] o2) {
                return Long.valueOf(o1[0]).compareTo(Long.valueOf(o2[0]));
            }

        });

        for (int i = 1; i < n; ++i) {
            cur += (pairs[i][0] - pairs[0][0]) * pairs[i][1];
        }
        long res = cur;
        long prefix = 0l;
        for (int i = 1; i < n; ++i) {
            prefix += pairs[i - 1][1];
            cur += (prefix * 2 - sumCost) * (pairs[i][0] - pairs[i - 1][0]);
            res = Math.min(res, cur);
        }
        return res;

    }

    // 6217. 使数组相似的最少操作次数
    public long makeSimilar(int[] nums, int[] target) {
        Arrays.sort(nums);
        Arrays.sort(target);
        int[] j = new int[2];
        long res = 0l;
        for (int num : nums) {
            int p = num % 2;
            while (target[j[p]] % 2 != p) {
                ++j[p];
            }
            res += Math.abs(num - target[j[p]++]);
        }
        return res / 4;

    }

    // 2354. 优质数对的数目 (Number of Excellent Pairs)
    public long countExcellentPairs(int[] nums, int k) {
        Set<Integer> visited = new HashSet<>();
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            if (visited.add(num)) {
                int bits = Integer.bitCount(num);
                map.put(bits, map.getOrDefault(bits, 0) + 1);
            }
        }
        long res = 0l;
        for (int key1 : map.keySet()) {
            for (int key2 : map.keySet()) {
                if (key1 + key2 >= k) {
                    res += map.get(key1) * map.get(key2);
                }
            }
        }
        return res;
    }

    // 948. 令牌放置 (Bag of Tokens)
    public int bagOfTokensScore(int[] tokens, int power) {
        int res = 0;
        Arrays.sort(tokens);
        int n = tokens.length;
        int i = 0;
        int j = n - 1;
        int points = 0;
        while (i <= j && (power >= tokens[i] || points > 0)) {
            while (i <= j && power >= tokens[i]) {
                ++points;
                power -= tokens[i++];
            }
            res = Math.max(res, points);
            if (i <= j && points > 0) {
                --points;
                power += tokens[j--];
            }
        }
        return res;

    }

    // 135. 分发糖果 (Candy)
    public int candy(int[] ratings) {
        int n = ratings.length;
        int[] left = new int[n];
        Arrays.fill(left, 1);
        int[] right = new int[n];
        Arrays.fill(right, 1);
        for (int i = 1; i < n; ++i) {
            if (ratings[i] > ratings[i - 1]) {
                left[i] = left[i - 1] + 1;
            }
        }
        for (int i = n - 2; i >= 0; --i) {
            if (ratings[i] > ratings[i + 1]) {
                right[i] = right[i + 1] + 1;
            }
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            res += Math.max(left[i], right[i]);
        }
        return res;
    }

    // LCP 67. 装饰树
    public TreeNode expandBinaryTree(TreeNode root) {
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (node.left != null) {
                    TreeNode added = new TreeNode(-1);
                    added.left = node.left;
                    node.left = added;
                    queue.offer(node.left.left);
                }

                if (node.right != null) {
                    TreeNode added = new TreeNode(-1);
                    added.right = node.right;
                    node.right = added;
                    queue.offer(node.right.right);
                }
            }
        }
        return root;

    }

    // LCP 67. 装饰树
    public TreeNode expandBinaryTree2(TreeNode root) {
        if (root == null || (root.left == null && root.right == null)) {
            return root;
        }
        if (root.left != null) {
            root.left = new TreeNode(-1, expandBinaryTree2(root.left), null);
        }
        if (root.right != null) {
            root.right = new TreeNode(-1, null, expandBinaryTree2(root.right));
        }
        return root;
    }

    // LCP 66. 最小展台数量
    public int minNumBooths(String[] demand) {
        int[] counts = new int[26];
        for (String d : demand) {
            int[] curCounts = new int[26];
            for (char c : d.toCharArray()) {
                ++curCounts[c - 'a'];
            }
            for (int i = 0; i < 26; ++i) {
                counts[i] = Math.max(counts[i], curCounts[i]);
            }
        }
        return Arrays.stream(counts).sum();

    }

    // 352. 将数据流变为多个不相交区间 (Data Stream as Disjoint Intervals) --并查集 还需掌握：二分查找
    class SummaryRanges {
        private Union352 union;
        private Set<Integer> set;

        public SummaryRanges() {
            union = new Union352(10002);
            set = new HashSet<>();
        }

        public void addNum(int value) {
            if (set.add(value)) {
                union.union(value, value + 1);
            }
        }

        public int[][] getIntervals() {
            List<int[]> res = new ArrayList<>();
            int i = 0;
            while (i <= 10000) {
                if (!set.contains(i)) {
                    ++i;
                    continue;
                }
                int j = union.getRoot(i);
                res.add(new int[] { i, j - 1 });
                i = j;
            }
            return res.toArray(new int[0][]);
        }
    }

    public class Union352 {
        private int[] rank;
        private int[] parent;

        public Union352(int n) {
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
            parent[root1] = root2;
        }
    }

    // 1156. 单字符重复子串的最大长度 (Swap For Longest Repeated Character Substring)
    public int maxRepOpt1(String text) {
        int[] counts = new int[26];
        for (char c : text.toCharArray()) {
            ++counts[c - 'a'];
        }
        int res = 0;
        int i = 0;
        int n = text.length();
        while (i < n) {
            char c = text.charAt(i);
            int len1 = 0;
            while (i + len1 < n && c == text.charAt(i + len1)) {
                ++len1;
            }
            int j = i + len1 + 1;
            int len2 = 0;
            while (j + len2 < n && c == text.charAt(j + len2)) {
                ++len2;
            }
            res = Math.max(res, Math.min(len1 + len2 + 1, counts[c - 'a']));
            i = j - 1;
        }
        return res;

    }

    // 1664. 生成平衡数组的方案数 (Ways to Make a Fair Array)
    public int waysToMakeFair(int[] nums) {
        int n = nums.length;
        int oddSum = 0;
        int evenSum = 0;
        for (int i = 0; i < n; ++i) {
            if ((i & 1) == 0) {
                evenSum += nums[i];
            } else {
                oddSum += nums[i];
            }
        }
        int sufOddSum = 0;
        int sufEvenSum = 0;
        int res = 0;
        for (int i = n - 1; i >= 0; --i) {
            int curOddSum = 0;
            int curEvenSum = 0;
            if ((i & 1) == 0) {
                curEvenSum = evenSum - nums[i] - sufEvenSum + sufOddSum;
                curOddSum = oddSum - sufOddSum + sufEvenSum;
                sufEvenSum += nums[i];
            } else {
                curOddSum = oddSum - nums[i] - sufOddSum + sufEvenSum;
                curEvenSum = evenSum - sufEvenSum + sufOddSum;
                sufOddSum += nums[i];
            }
            if (curOddSum == curEvenSum) {
                ++res;
            }
        }
        return res;

    }

    // 2069. 模拟行走机器人 II (Walking Robot Simulation II)
    class Robot {
        private int width;
        private int height;
        private String[] dirStrings = { "East", "North", "West", "South" };
        private int d;
        private int x;
        private int y;
        private int circleSteps;

        public Robot(int width, int height) {
            this.width = width;
            this.height = height;
            circleSteps = (height + width) * 2 - 4;
        }

        public void step(int num) {
            num %= circleSteps;
            while (num != 0) {
                int steps = 0;
                switch (d) {
                    case 0:
                        steps = Math.min(num, width - x - 1);
                        x += steps;
                        break;
                    case 1:
                        steps = Math.min(num, height - y - 1);
                        y += steps;
                        break;
                    case 2:
                        steps = Math.min(num, x);
                        x -= steps;
                        break;
                    case 3:
                        steps = Math.min(num, y);
                        y -= steps;
                        break;
                }
                num -= steps;
                if (num > 0) {
                    d = (d + 1) % 4;
                }
            }
            if (x == 0 && y == 0) {
                d = 3;
            }
        }

        public int[] getPos() {
            return new int[] { x, y };
        }

        public String getDir() {
            return dirStrings[d];
        }
    }

    // 1761. 一个图中连通三元组的最小度数 (Minimum Degree of a Connected Trio in a Graph)
    public int minTrioDegree(int n, int[][] edges) {
        boolean[][] connected = new boolean[n][n];
        int[] degrees = new int[n];
        for (int[] edge : edges) {
            connected[edge[0] - 1][edge[1] - 1] = true;
            connected[edge[1] - 1][edge[0] - 1] = true;
            ++degrees[edge[0] - 1];
            ++degrees[edge[1] - 1];
        }
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                if (!connected[i][j]) {
                    continue;
                }
                for (int k = 0; k < n; ++k) {
                    if (connected[i][k] && connected[j][k]) {
                        res = Math.min(res, degrees[i] + degrees[j] + degrees[k] - 6);
                    }
                }
            }
        }
        return res == Integer.MAX_VALUE ? -1 : res;

    }

    // 1878. 矩阵中最大的三个菱形和 (Get Biggest Three Rhombus Sums in a Grid) --枚举中心点
    public int[] getBiggestThree(int[][] grid) {
        int first = 0;
        int second = 0;
        int third = 0;
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int cur = grid[i][j];
                for (int side = 0; side <= Math.min(Math.min(j, n - 1 - j), Math.min(i, m - 1 - i)); ++side) {
                    if (side != 0) {
                        int leftX = i;
                        int leftY = j - side;
                        int topX = i - side;
                        int topY = j;
                        int rightX = i;
                        int rightY = j + side;
                        int bottomX = i + side;
                        int bottomY = j;
                        for (int offset = 0; offset < side; ++offset) {
                            cur += grid[leftX--][leftY++];
                            cur += grid[topX++][topY++];
                            cur += grid[rightX++][rightY--];
                            cur += grid[bottomX--][bottomY--];
                        }
                    }
                    if (cur > first) {
                        third = second;
                        second = first;
                        first = cur;
                    } else if (cur > second && cur < first) {
                        third = second;
                        second = cur;
                    } else if (cur > third && cur < second) {
                        third = cur;
                    }
                    cur = 0;
                }
            }
        }
        if (first == 0) {
            return new int[0];
        }
        if (second == 0) {
            return new int[] { first };
        }
        if (third == 0) {
            return new int[] { first, second };
        }
        return new int[] { first, second, third };

    }

    // 2034. 股票价格波动 (Stock Price Fluctuation)
    class StockPrice {
        private TreeMap<Integer, Integer> timeToPrice;
        private TreeMap<Integer, Integer> priceToCnts;

        public StockPrice() {
            this.timeToPrice = new TreeMap<>();
            this.priceToCnts = new TreeMap<>();

        }

        public void update(int timestamp, int price) {
            int pre = timeToPrice.getOrDefault(timestamp, 0);
            if (pre != 0) {
                priceToCnts.merge(pre, -1, Integer::sum);
                if (priceToCnts.get(pre) <= 0) {
                    priceToCnts.remove(pre);
                }
            }
            timeToPrice.put(timestamp, price);
            priceToCnts.merge(price, 1, Integer::sum);
        }

        public int current() {
            return timeToPrice.lastEntry().getValue();
        }

        public int maximum() {
            return priceToCnts.lastKey();

        }

        public int minimum() {
            return priceToCnts.firstKey();
        }
    }

    // 1513. 仅含 1 的子串数 (Number of Substrings With Only 1s)
    public int numSub(String s) {
        final int MOD = (int) (1e9 + 7);
        int res = 0;
        int cnt1 = 0;
        for (char c : s.toCharArray()) {
            if (c == '1') {
                res += ++cnt1;
                res %= MOD;
            } else {
                cnt1 = 0;
            }
        }
        return res;

    }

    // 2451. 差值数组不同的字符串 (Odd String Difference)
    public String oddString(String[] words) {
        int val0 = 0;
        int n = words.length;
        int index = -1;
        for (int i = 0; i < n; ++i) {
            int cur = getNum2451(words[i]);
            if (i == 0) {
                val0 = cur;
            } else if (cur != val0) {
                index = i;
                break;
            }
        }
        if (index > 1) {
            return words[index];
        }
        int val2 = getNum2451(words[2]);
        if (val0 != val2) {
            return words[0];
        }
        return words[1];

    }

    private int getNum2451(String s) {
        final int BASE = 53;
        final int MOD = 401;
        char[] arr = s.toCharArray();
        int n = arr.length;
        int res = 0;
        int mul = 1;
        for (int j = 0; j < n - 1; ++j) {
            int num = arr[j + 1] - arr[j] + 26;
            res = (res + num * mul) % MOD;
            mul = mul * BASE % MOD;
        }
        return res;
    }

    // 6228. 距离字典两次编辑以内的单词
    public List<String> twoEditWords(String[] queries, String[] dictionary) {
        List<String> res = new ArrayList<>();
        for (String query : queries) {
            if (check6228(query, dictionary)) {
                res.add(query);
            }
        }
        return res;

    }

    private boolean check6228(String query, String[] dictionary) {
        search: for (String dic : dictionary) {
            int count = 0;
            for (int i = 0; i < dic.length(); ++i) {
                if (dic.charAt(i) != query.charAt(i)) {
                    ++count;
                    if (count > 2) {
                        continue search;
                    }
                }
            }
            return true;
        }
        return false;
    }

    // 6226. 摧毁一系列目标
    public int destroyTargets(int[] nums, int space) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num % space, map.getOrDefault(num % space, 0) + 1);
        }
        int maxSize = 0;
        int res = Integer.MAX_VALUE;
        for (int num : nums) {
            if (map.get(num % space) > maxSize) {
                maxSize = map.get(num % space);
                res = num;
            } else if (map.get(num % space) == maxSize) {
                res = Math.min(res, num);
            }
        }
        return res;

    }

    // 6220. 可被三整除的偶数的平均值
    public int averageValue(int[] nums) {
        int count = 0;
        int sum = 0;
        for (int num : nums) {
            if (num % 6 == 0) {
                ++count;
                sum += num;
            }
        }
        if (count == 0) {
            return 0;

        }
        return sum / count;

    }

    // 2456. 最流行的视频创作者
    public List<List<String>> mostPopularCreator(String[] creators, String[] ids, int[] views) {
        int n = creators.length;
        Map<String, TreeSet<Bean2456>> map = new HashMap<>();
        Map<String, Long> maxMap = new HashMap<>();
        long maxSum = 0l;
        for (int i = 0; i < n; ++i) {
            map.computeIfAbsent(creators[i], k -> new TreeSet<>()).add(new Bean2456(ids[i], views[i]));
            maxMap.put(creators[i], maxMap.getOrDefault(creators[i], 0l) + views[i]);
            maxSum = Math.max(maxSum, maxMap.get(creators[i]));
        }
        List<List<String>> res = new ArrayList<>();
        for (Map.Entry<String, TreeSet<Bean2456>> entry : map.entrySet()) {
            if (maxMap.get(entry.getKey()) == maxSum) {
                res.add(List.of(entry.getKey(), entry.getValue().first().id));
            }
        }
        return res;

    }

    class Bean2456 implements Comparable<Bean2456> {
        String id;
        int score;

        public Bean2456(String id, int score) {
            this.id = id;
            this.score = score;
        }

        @Override
        public int compareTo(Bean2456 o) {
            return this.score == o.score ? this.id.compareTo(o.id) : o.score - this.score;
        }
    }

    // 6222. 美丽整数的最小增量
    public long makeIntegerBeautiful(long n, int target) {
        int sum = getSum6222(n);
        if (sum <= target) {
            return 0l;
        }
        long d = 10l;
        long copy = n;
        while (true) {
            long cur = ((copy / d) + 1) * d;
            if (getSum6222(cur) <= target) {
                return cur - n;
            }
            copy = cur;
            d *= 10;

        }

    }

    private int getSum6222(long n) {
        int sum = 0;
        while (n != 0) {
            sum += n % 10;
            n /= 10;
        }
        return sum;
    }

    // 1604. 警告一小时内使用相同员工卡大于等于三次的人 (Alert Using Same Key-Card Three or More Times in
    // a One Hour)
    public List<String> alertNames(String[] keyName, String[] keyTime) {
        int n = keyName.length;
        List<String> res = new ArrayList<>();
        Map<String, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            int h = Integer.parseInt(keyTime[i].substring(0, 2));
            int m = Integer.parseInt(keyTime[i].substring(3));
            map.computeIfAbsent(keyName[i], k -> new ArrayList<>()).add(h * 60 + m);
        }
        search: for (Map.Entry<String, List<Integer>> entry : map.entrySet()) {
            List<Integer> t = entry.getValue();
            Collections.sort(t);
            for (int i = 2; i < t.size(); ++i) {
                if (t.get(i) - t.get(i - 2) <= 60) {
                    res.add(entry.getKey());
                    continue search;
                }
            }
        }
        Collections.sort(res);
        return res;

    }

    // 1573. 分割字符串的方案数 (Number of Ways to Split a String)
    public int numWays(String s) {
        int n = s.length();
        int countOne = 0;
        for (int i = 0; i < n; ++i) {
            countOne += s.charAt(i) - '0';
        }
        if (countOne % 3 != 0) {
            return 0;
        }
        final int mod = (int) (1e9 + 7);
        if (countOne == 0) {
            return (int) (((long) n - 1) * (n - 2) / 2 % mod);
        }
        int count1 = -1;
        int count2 = -1;
        int count = 0;
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) == '1') {
                ++count;
                if (count == countOne / 3) {
                    count1 = i;
                } else if (count == countOne / 3 + 1) {
                    count1 = i - count1;
                }
                if (count == countOne / 3 * 2) {
                    count2 = i;
                } else if (count == countOne / 3 * 2 + 1) {
                    count2 = i - count2;
                    break;
                }
            }
        }
        return (int) ((long) count1 * count2 % mod);

    }

    // 1662. 检查两个字符串数组是否相等 (Check If Two String Arrays are Equivalent)
    public boolean arrayStringsAreEqual(String[] word1, String[] word2) {
        int i1 = 0;
        int i2 = 0;
        int j1 = 0;
        int j2 = 0;
        while (i1 < word1.length && i2 < word2.length) {
            String w1 = word1[i1];
            String w2 = word2[i2];

            if (w1.charAt(j1++) != w2.charAt(j2++)) {
                return false;
            }
            if (j1 == w1.length()) {
                j1 = 0;
                ++i1;
            }
            if (j2 == w2.length()) {
                j2 = 0;
                ++i2;
            }

        }
        return i1 == word1.length && i2 == word2.length;

    }

    // 2012. 数组美丽值求和 (Sum of Beauty in the Array)
    public int sumOfBeauties(int[] nums) {
        int n = nums.length;
        boolean[] right = new boolean[n];
        int mx = nums[n - 1];
        for (int i = n - 2; i >= 1; --i) {
            if (nums[i] < mx) {
                right[i] = true;
                mx = nums[i];
            }
        }
        mx = nums[0];
        int res = 0;
        for (int i = 1; i < n - 1; ++i) {
            if (mx < nums[i] && right[i]) {
                res += 2;
            } else if (nums[i - 1] < nums[i] && nums[i] < nums[i + 1]) {
                res += 1;
            }
            mx = Math.max(mx, nums[i]);
        }
        return res;

    }

    // 2012. 数组美丽值求和 (Sum of Beauty in the Array)
    public int sumOfBeauties2(int[] nums) {
        int n = nums.length;
        int[] left = new int[n];
        for (int i = 1; i < n; ++i) {
            left[i] = Math.max(left[i - 1], nums[i - 1]);
        }
        int res = 0;
        int rightMin = Integer.MAX_VALUE;
        for (int i = n - 2; i >= 1; --i) {
            rightMin = Math.min(rightMin, nums[i + 1]);
            if (left[i] < nums[i] && nums[i] < rightMin) {
                res += 2;
            } else if (nums[i - 1] < nums[i] && nums[i] < nums[i + 1]) {
                res += 1;
            }
        }
        return res;
    }

    // 1968. 构造元素不等于两相邻元素平均值的数组 (Array With Elements Not Equal to Average of
    // Neighbors)
    public int[] rearrangeArray(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        int[] res = new int[n];
        for (int i = 0; i < (n + 1) / 2; ++i) {
            res[i * 2] = nums[i];
            if (i * 2 + 1 < n) {
                res[i * 2 + 1] = nums[(n + 1) / 2 + i];
            }

        }
        return res;

    }

    // 1946. 子字符串突变后可能得到的最大整数 (Largest Number After Mutating Substring)
    public String maximumNumber(String num, int[] change) {
        char[] chars = num.toCharArray();
        int n = chars.length;
        int i = 0;
        while (i < n) {
            int cur = chars[i] - '0';
            if (cur < change[cur]) {
                break;
            }
            ++i;
        }

        while (i < n) {
            int cur = chars[i] - '0';
            if (cur > change[cur]) {
                break;
            }
            chars[i] = (char) (change[cur] + '0');
            ++i;
        }
        return String.valueOf(chars);

    }

    // 1705. 吃苹果的最大数目 (Maximum Number of Eaten Apples)
    public int eatenApples(int[] apples, int[] days) {
        int n = apples.length;
        int res = 0;
        PriorityQueue<int[]> q = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[1], o2[1]);
            }

        });
        for (int i = 0; i < n || !q.isEmpty(); ++i) {
            if (i < n) {
                q.offer(new int[] { apples[i], i + days[i] });
            }
            while (!q.isEmpty() && (q.peek()[0] == 0 || q.peek()[1] <= i)) {
                q.poll();
            }
            if (!q.isEmpty()) {
                ++res;
                --q.peek()[0];
            }
        }
        return res;

    }

    // 给定一个只由小写字母和数字组成的字符串，要求子串必须只有一个小写字母，求这样的子串的最大长度
    public int getMaxLengthWithOneLetter(String s) {
        List<Integer> list = new ArrayList<>();
        int n = s.length();
        list.add(-1);
        for (int i = 0; i < n; ++i) {
            if (Character.isLetter(s.charAt(i))) {
                list.add(i);
            }
        }
        list.add(n);
        int res = 0;
        for (int i = 2; i < list.size(); ++i) {
            res = Math.max(res, list.get(i) - list.get(i - 2) - 1);
        }
        return res;
    }

    // 给定一个只由小写字母和数字组成的字符串，要求子串必须「最多」只有一个小写字母，求这样的子串的最大长度
    public int getMaxLengthWithMostOneLetter(String s) {
        int n = s.length();
        int left = 0;
        int right = 0;
        int letterCount = 0;
        int res = 0;
        while (right < n) {
            if (Character.isLetter(s.charAt(right))) {
                ++letterCount;
            }
            while (letterCount >= 2) {
                if (Character.isLetter(s.charAt(left))) {
                    --letterCount;
                }
                ++left;
            }
            res = Math.max(res, right - left + 1);
            ++right;
        }
        return res;

    }

    // 966. 元音拼写检查器 (Vowel Spellchecker)
    public String[] spellchecker(String[] wordlist, String[] queries) {
        int u = 0;
        for (char c : "aeiou".toCharArray()) {
            u |= 1 << (c - 'a');
        }
        Set<String> s = new HashSet<>();
        Map<String, Integer> lowerMap = new HashMap<>();
        Map<String, Integer> vowelMap = new HashMap<>();
        for (int i = 0; i < wordlist.length; ++i) {
            s.add(wordlist[i]);
            String lower = wordlist[i].toLowerCase();
            lowerMap.putIfAbsent(lower, i);
            char[] a = lower.toCharArray();
            for (int j = 0; j < a.length; ++j) {
                if (isVowel(u, a[j])) {
                    a[j] = '_';
                }
            }
            String v = String.valueOf(a);
            vowelMap.putIfAbsent(v, i);
        }
        for (int i = 0; i < queries.length; ++i) {
            String cur = queries[i];
            if (s.contains(cur)) {
                continue;
            }
            String lower = queries[i].toLowerCase();
            if (lowerMap.containsKey(lower)) {
                queries[i] = wordlist[lowerMap.get(lower)];
                continue;
            }
            char[] a = lower.toCharArray();
            for (int j = 0; j < a.length; ++j) {
                if (isVowel(u, a[j])) {
                    a[j] = '_';
                }
            }
            String v = String.valueOf(a);
            if (vowelMap.containsKey(v)) {
                queries[i] = wordlist[vowelMap.get(v)];
                continue;
            }
            queries[i] = "";
        }
        return queries;

    }

    private boolean isVowel(int u, char c) {
        return ((u >> (c - 'a')) & 1) != 0;
    }

    // 1390. 四因数 (Four Divisors)
    public int sumFourDivisors(int[] nums) {
        int max = 0;
        for (int num : nums) {
            max = Math.max(max, num);
        }
        int[] arr = new int[max + 1];
        int res = 0;
        for (int num : nums) {
            if (arr[num] == -1) {
                continue;
            }
            if (arr[num] > 0) {
                res += arr[num];
                continue;
            }
            int sum = check1390(num);
            arr[num] = sum;
            if (arr[num] > 0) {
                res += sum;
            }
        }
        return res;

    }

    private int check1390(int num) {
        int count = 0;
        int sum = 0;
        int i = 1;
        while (i * i <= num) {
            if (num % i == 0) {
                if (++count > 4) {
                    return -1;
                }
                sum += i;
                if (num / i != i) {
                    if (++count > 4) {
                        return -1;
                    }
                    sum += num / i;
                }
            }
            ++i;
        }
        return count == 4 ? sum : -1;
    }

    // 2007. 从双倍数组中还原原数组 (Find Original Array From Doubled Array)
    public int[] findOriginalArray(int[] changed) {
        int n = changed.length;
        if (n % 2 == 1) {
            return new int[0];
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int c : changed) {
            map.merge(c, 1, Integer::sum);
        }
        int cnt0 = map.getOrDefault(0, 0);
        if (cnt0 % 2 == 1) {
            return new int[0];
        }
        int[] res = new int[n / 2];
        Arrays.sort(changed);
        int index = cnt0 / 2;
        for (int i = cnt0; i < n; ++i) {
            if (map.getOrDefault(changed[i], 0) == 0) {
                continue;
            }
            if (map.getOrDefault(changed[i] * 2, 0) == 0) {
                return new int[0];
            }
            map.merge(changed[i], -1, Integer::sum);
            map.merge(changed[i] * 2, -1, Integer::sum);
            if (map.getOrDefault(changed[i], 0) == 0) {
                map.remove(changed[i]);
            }
            if (map.getOrDefault(changed[i] * 2, 0) == 0) {
                map.remove(changed[i] * 2);
            }
            res[index++] = changed[i];
        }
        return res;
    }

    // 686. 重复叠加字符串匹配 (Repeated String Match)
    public int repeatedStringMatch(String a, String b) {
        int res = 1;
        StringBuilder builder = new StringBuilder(a);
        while (builder.length() < b.length()) {
            builder.append(a);
            ++res;
        }
        if (builder.toString().indexOf(b) != -1) {
            return res;
        }
        builder.append(a);
        ++res;
        if (builder.toString().indexOf(b) != -1) {
            return res;
        }
        return -1;

    }

    // LCP 68. 美观的花束
    public int beautifulBouquet(int[] flowers, int cnt) {
        int n = flowers.length;
        int max = 0;
        for (int flower : flowers) {
            max = Math.max(max, flower);
        }
        int[] counts = new int[max + 1];
        int left = 0;
        int right = 0;
        long res = 0l;
        final int mod = (int) (1e9 + 7);
        while (right < n) {
            ++counts[flowers[right]];
            while (counts[flowers[right]] > cnt) {
                --counts[flowers[left++]];
            }
            res = (res + right - left + 1) % mod;
            ++right;
        }
        return (int) (res % mod);

    }

    // 2157. 字符串分组 (Groups of Strings)
    public int[] groupStrings(String[] words) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (String word : words) {
            int mask = 0;
            for (char c : word.toCharArray()) {
                mask |= 1 << (c - 'a');
            }
            counts.put(mask, counts.getOrDefault(mask, 0) + 1);
        }
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue = new LinkedList<>();
        int max = 0;
        int groups = 0;
        for (Map.Entry<Integer, Integer> entry : counts.entrySet()) {
            if (visited.contains(entry.getKey())) {
                continue;
            }
            int curTotal = entry.getValue();
            queue.offer(entry.getKey());
            visited.add(entry.getKey());
            while (!queue.isEmpty()) {
                int cur = queue.poll();
                for (int neighbor : getNeighbors2157(cur)) {
                    if (counts.containsKey(neighbor) && !visited.contains(neighbor)) {
                        curTotal += counts.get(neighbor);
                        visited.add(neighbor);
                        queue.offer(neighbor);
                    }
                }
            }
            max = Math.max(max, curTotal);
            ++groups;
        }
        return new int[] { groups, max };

    }

    private List<Integer> getNeighbors2157(int mask) {
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < 26; ++i) {
            res.add(mask ^ (1 << i));
        }
        for (int i = 0; i < 26; ++i) {
            if ((mask & (1 << i)) == 0) {
                for (int j = 0; j < 26; ++j) {
                    if ((mask & (1 << j)) != 0) {
                        res.add(mask ^ (1 << i) ^ (1 << j));
                    }
                }
            }
        }
        return res;
    }

    // 2157. 字符串分组 (Groups of Strings)
    public int[] groupStrings2(String[] words) {
        Map<Integer, Integer> cntMap = new HashMap<>();
        for (String word : words) {
            int m = getMask2157(word);
            cntMap.merge(m, 1, Integer::sum);
        }
        int index = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int key : cntMap.keySet()) {
            map.put(key, index++);
        }
        Union2157 union = new Union2157(index);
        for (String word : words) {
            int m = getMask2157(word);
            int i = map.get(m);
            // 添加
            for (int k = 0; k < 26; ++k) {
                int c = m | (1 << k);
                if (map.containsKey(c)) {
                    union.union(i, map.get(c));
                }
            }

            // 替换
            int c = m;
            while (c != 0) {
                int j = Integer.numberOfTrailingZeros(c);
                int k = m ^ (1 << j);
                // 删除
                if (map.containsKey(k)) {
                    union.union(i, map.get(k));
                }
                for (int s = 0; s < 26; ++s) {
                    int co = k | (1 << s);
                    if (map.containsKey(co)) {
                        union.union(i, map.get(co));
                    }
                }
                c &= c - 1;
            }
        }
        int[] res = new int[2];
        res[0] = union.getCnt();
        Map<Integer, Integer> cc = new HashMap<>();
        for (Map.Entry<Integer, Integer> entry : cntMap.entrySet()) {
            int m = entry.getKey();
            int i = map.get(m);
            int root = union.getRoot(i);
            cc.merge(root, entry.getValue(), Integer::sum);
        }
        res[1] = Collections.max(cc.values());
        return res;

    }

    private int getMask2157(String word) {
        int m = 0;
        for (char c : word.toCharArray()) {
            m |= 1 << (c - 'a');
        }
        return m;
    }

    public class Union2157 {
        private int[] rank;
        private int[] parent;
        private int cnt;

        public Union2157(int n) {
            this.rank = new int[n];
            this.parent = new int[n];
            for (int i = 0; i < n; ++i) {
                rank[i] = 1;
                parent[i] = i;
            }
            this.cnt = n;
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
            --cnt;

        }

        public int getCnt() {
            return cnt;
        }

    }

    // 1234. 替换子串得到平衡字符串 (Replace the Substring for Balanced String) --滑动窗口
    public int balancedString(String s) {
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'A'];
        }
        int n = s.length();
        int per = n / 4;
        counts['Q' - 'A'] -= per;
        counts['W' - 'A'] -= per;
        counts['E' - 'A'] -= per;
        counts['R' - 'A'] -= per;
        int i = 0;
        int j = 0;
        int[] cur = new int[26];
        int res = n;
        while (j < n) {
            ++cur[s.charAt(j) - 'A'];
            while (i < n && cur['Q' - 'A'] >= counts['Q' - 'A'] && cur['W' - 'A'] >= counts['W' - 'A']
                    && cur['E' - 'A'] >= counts['E' - 'A'] && cur['R' - 'A'] >= counts['R' - 'A']) {
                res = Math.min(res, j - i + 1);
                --cur[s.charAt(i++) - 'A'];
            }
            ++j;
        }
        return res;

    }

    // 1895. 最大的幻方 (Largest Magic Square)
    public int largestMagicSquare(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] prefixRow = new int[m][n + 1];
        for (int i = 0; i < m; ++i) {
            for (int j = 1; j < n + 1; ++j) {
                prefixRow[i][j] = prefixRow[i][j - 1] + grid[i][j - 1];
            }
        }
        int[][] prefixCol = new int[m + 1][n];
        for (int j = 0; j < n; ++j) {
            for (int i = 1; i < m + 1; ++i) {
                prefixCol[i][j] = prefixCol[i - 1][j] + grid[i - 1][j];
            }
        }
        // main diagonal
        int[][] mainDiagonal = new int[m + 1][n + 1];
        for (int i = 1; i < m + 1; ++i) {
            for (int j = 1; j < n + 1; ++j) {
                mainDiagonal[i][j] = mainDiagonal[i - 1][j - 1] + grid[i - 1][j - 1];
            }
        }

        for (int side = Math.min(m, n); side >= 2; --side) {
            for (int i = 0; i + side - 1 < m; ++i) {
                search: for (int j = 0; j + side - 1 < n; ++j) {
                    int val = prefixRow[i][j + side] - prefixRow[i][j];
                    for (int x = i + 1; x < i + side; ++x) {
                        if (prefixRow[x][j + side] - prefixRow[x][j] != val) {
                            continue search;
                        }
                    }
                    for (int y = j; y < j + side; ++y) {
                        if (prefixCol[i + side][y] - prefixCol[i][y] != val) {
                            continue search;
                        }
                    }
                    if (mainDiagonal[i + side][j + side] - mainDiagonal[i][j] != val) {
                        continue search;
                    }
                    int counterSum = 0;
                    int y = j + side - 1;
                    for (int x = i; x < i + side; ++x) {
                        counterSum += grid[x][y];
                        --y;
                    }
                    if (counterSum != val) {
                        continue search;
                    }
                    return side;
                }
            }
        }
        return 1;

    }

    // 981. 基于时间的键值存储 (Time Based Key-Value Store) --二分查找
    class TimeMap {
        class Bean {
            String value;
            int timeStamp;

            Bean(String value, int timeStamp) {
                this.value = value;
                this.timeStamp = timeStamp;

            }
        }

        private Map<String, List<Bean>> map;

        public TimeMap() {
            map = new HashMap<>();
        }

        public void set(String key, String value, int timestamp) {
            map.computeIfAbsent(key, k -> new ArrayList<>()).add(new Bean(value, timestamp));
        }

        public String get(String key, int timestamp) {
            if (!map.containsKey(key)) {
                return "";
            }
            List<Bean> list = map.get(key);
            int index = binarySearch981(list, timestamp);
            if (index == -1) {
                return "";
            }
            return list.get(index).value;

        }

        private int binarySearch981(List<Bean> list, int timestamp) {
            int n = list.size();
            if (timestamp >= list.get(n - 1).timeStamp) {
                return n - 1;
            }
            if (timestamp < list.get(0).timeStamp) {
                return -1;
            }
            int left = 0;
            int right = n - 1;
            int res = -1;
            while (left <= right) {
                int mid = left + ((right - left) >>> 1);
                if (list.get(mid).timeStamp <= timestamp) {
                    res = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return res;
        }
    }

    // 1124. 表现良好的最长时间段 (Longest Well-Performing Interval)
    public int longestWPI(int[] hours) {
        Map<Integer, Integer> map = new HashMap<>();
        int n = hours.length;
        int prefix = 0;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            prefix += hours[i] > 8 ? 1 : -1;
            if (prefix > 0) {
                res = Math.max(res, i + 1);
            } else if (map.containsKey(prefix - 1)) {
                res = Math.max(res, i - map.get(prefix - 1));
            }
            map.putIfAbsent(prefix, i);
        }
        return res;

    }

    // 1678. 设计 Goal 解析器 (Goal Parser Interpretation)
    public String interpret(String command) {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < command.length(); ++i) {
            char c = command.charAt(i);
            if (c == 'G') {
                res.append(c);
            } else if (c == ')') {
                if (command.charAt(i - 1) == '(') {
                    res.append('o');
                } else {
                    res.append("al");
                }
            }
        }
        return res.toString();

    }

    // 2460. 对数组执行操作 (Apply Operations to an Array)
    public int[] applyOperations(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n - 1; ++i) {
            if (nums[i] == nums[i + 1]) {
                nums[i] *= 2;
                nums[i + 1] = 0;
            }
        }
        int[] res = new int[n];
        int i = 0;
        int j = 0;
        while (i < n) {
            if (nums[i] != 0) {
                res[j++] = nums[i];
            }
            ++i;
        }
        return res;

    }

    // 2461. 长度为 K 子数组中的最大和 (Maximum Sum of Distinct Subarrays With Length K)
    public long maximumSubarraySum(int[] nums, int k) {
        long res = 0L;
        long s = 0L;
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            s += nums[i];
            map.merge(nums[i], 1, Integer::sum);
            if (i >= k) {
                s -= nums[i - k];
                map.merge(nums[i - k], -1, Integer::sum);
                if (map.get(nums[i - k]) == 0) {
                    map.remove(nums[i - k]);
                }
            }
            if (i >= k - 1 && map.size() == k) {
                res = Math.max(res, s);
            }
        }
        return res;

    }

    // 6231. 雇佣 K 位工人的总代价
    public long totalCost(int[] costs, int k, int candidates) {
        int n = costs.length;
        long res = 0l;
        if (candidates * 2 >= n) {
            Arrays.sort(costs);
            for (int i = 0; i < k; ++i) {
                res += costs[i];
            }
            return res;
        }
        int count = n - candidates * 2;
        Queue<Integer> queue1 = new PriorityQueue<>();
        Queue<Integer> queue2 = new PriorityQueue<>();
        for (int i = 0; i < candidates; ++i) {
            queue1.offer(costs[i]);
            queue2.offer(costs[n - i - 1]);
        }
        int index1 = candidates;
        int index2 = n - candidates - 1;
        while (count > 0 && k > 0) {
            if (queue1.peek() <= queue2.peek()) {
                res += queue1.poll();
                queue1.offer(costs[index1++]);
            } else {
                res += queue2.poll();
                queue2.offer(costs[index2--]);
            }
            --count;
            --k;
        }
        if (k == 0) {
            return res;
        }
        queue1.addAll(queue2);
        while (k-- > 0) {
            res += queue1.poll();
        }
        return res;
    }

    // 1288. 删除被覆盖区间 (Remove Covered Intervals)
    public int removeCoveredIntervals(int[][] intervals) {
        int n = intervals.length;
        int res = intervals.length;
        Arrays.sort(intervals, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return o2[1] - o1[1];
                }
                return o1[0] - o2[0];
            }

        });

        int rightMax = intervals[0][1];
        for (int i = 1; i < n; ++i) {
            if (intervals[i][1] <= rightMax) {
                --res;
            }
            rightMax = Math.max(rightMax, intervals[i][1]);
        }
        return res;

    }

    // 1589. 所有排列中的最大和 (Maximum Sum Obtained of Any Permutation)
    public int maxSumRangeQuery(int[] nums, int[][] requests) {
        int n = nums.length;
        int[] diff = new int[n + 1];
        for (int[] request : requests) {
            ++diff[request[0]];
            --diff[request[1] + 1];
        }
        for (int i = 1; i < n + 1; ++i) {
            diff[i] += diff[i - 1];
        }
        int[][] pairs = new int[n][2];
        for (int i = 0; i < n; ++i) {
            pairs[i][0] = i;
            pairs[i][1] = diff[i];
        }

        Arrays.sort(pairs, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o2[1] - o1[1];
            }

        });

        Arrays.sort(nums);

        int[] arr = new int[n];
        for (int i = 0; i < n; ++i) {
            arr[pairs[i][0]] = nums[n - i - 1];
        }
        long[] prefix = new long[n + 1];
        for (int i = 1; i < n + 1; ++i) {
            prefix[i] = prefix[i - 1] + arr[i - 1];
        }
        long res = 0l;
        final int mod = (int) (1e9 + 7);
        for (int[] request : requests) {
            res = (res + prefix[request[1] + 1] - prefix[request[0]]) % mod;
        }
        return (int) res;

    }

    // 1589. 所有排列中的最大和 (Maximum Sum Obtained of Any Permutation)
    public int maxSumRangeQuery2(int[] nums, int[][] requests) {
        int n = nums.length;
        int[] diff = new int[n];
        for (int[] request : requests) {
            ++diff[request[0]];
            if (request[1] + 1 < n) {
                --diff[request[1] + 1];
            }
        }
        for (int i = 1; i < n; ++i) {
            diff[i] += diff[i - 1];
        }

        Arrays.sort(nums);
        Arrays.sort(diff);

        long res = 0l;
        final int mod = (int) (1e9 + 7);

        for (int i = n - 1; i >= 0 && diff[i] > 0; --i) {
            res = (res + (long) nums[i] * diff[i]) % mod;
        }
        return (int) res;

    }

    // 880. 索引处的解码字符串 (Decoded String at Index)
    public String decodeAtIndex(String s, int k) {
        int n = s.length();
        int i = 0;
        long count = 0l;
        while (i < n) {
            char c = s.charAt(i);
            if (Character.isDigit(c)) {
                count *= c - '0';
            } else {
                ++count;
            }
            if (count >= k) {
                break;
            }
            ++i;
        }
        while (i >= 0) {
            char c = s.charAt(i);
            k %= count;
            if (k == 0 && Character.isLetter(c)) {
                return String.valueOf(c);
            }
            if (Character.isDigit(c)) {
                count /= c - '0';
            } else {
                --count;
            }
            --i;
        }
        return "";

    }

    // 1015. 可被 K 整除的最小整数 (Smallest Integer Divisible by K)
    public int smallestRepunitDivByK(int k) {
        if (k % 2 == 0 || k % 5 == 0) {
            return -1;
        }
        int res = 1;
        int v = 1 % k;
        while (v != 0) {
            v = (v * 10 + 1) % k;
            ++res;
        }
        return res;

    }

    // 1942. 最小未被占据椅子的编号 (The Number of the Smallest Unoccupied Chair)
    public int smallestChair(int[][] times, int targetFriend) {
        int n = times.length;
        Queue<Integer> queue = new PriorityQueue<>();
        int[][] arrive = new int[n][2];
        int[][] leave = new int[n][2];
        for (int i = 0; i < n; ++i) {
            queue.offer(i);
            arrive[i][0] = times[i][0];
            arrive[i][1] = i;
            leave[i][0] = times[i][1];
            leave[i][1] = i;
        }
        Arrays.sort(arrive, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });
        Arrays.sort(leave, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }

        });
        Map<Integer, Integer> map = new HashMap<>();
        int j = 0;
        for (int[] arr : arrive) {
            while (j < n && leave[j][0] <= arr[0]) {
                queue.offer(map.get(leave[j][1]));
                ++j;
            }
            map.put(arr[1], queue.poll());
            if (arr[1] == targetFriend) {
                return map.get(arr[1]);
            }
        }
        return -1;

    }

    // 1108. IP 地址无效化 (Defanging an IP Address)
    public String defangIPaddr(String address) {
        StringBuilder res = new StringBuilder();
        for (char c : address.toCharArray()) {
            if (c == '.') {
                res.append("[.]");
            } else {
                res.append(c);
            }
        }
        return res.toString();

    }

    // 249. 移位字符串分组 (Group Shifted Strings) --plus
    public List<List<String>> groupStrings249(String[] strings) {
        List<List<String>> res = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for (String string : strings) {
            char[] chars = string.toCharArray();
            int shifts = chars[0] - 'a';
            for (int i = 0; i < chars.length; ++i) {
                chars[i] = (char) ((chars[i] - shifts + 26) % 26);
            }
            String s = String.valueOf(chars);
            map.computeIfAbsent(s, k -> new ArrayList<>()).add(string);
        }
        for (List<String> list : map.values()) {
            res.add(list);
        }
        return res;

    }

    // 254. 因子的组合 (Factor Combinations) --plus
    public List<List<Integer>> getFactors(int n) {
        List<List<Integer>> res = new ArrayList<>();
        dfs254(n, 2, res, new ArrayList<>());
        return res;

    }

    private void dfs254(int n, int i, List<List<Integer>> res, List<Integer> cur) {
        if (!cur.isEmpty()) {
            cur.add(n);
            res.add(new ArrayList<>(cur));
            cur.remove(cur.size() - 1);
        }
        for (int x = i; x * x <= n; ++x) {
            if (n % x == 0) {
                cur.add(x);
                dfs254(n / x, x, res, cur);
                cur.remove(cur.size() - 1);
            }
        }
    }

    // 694. 不同岛屿的数量 (Number of Distinct Islands) --plus
    public int numDistinctIslands(int[][] grid) {
        Set<String> set = new HashSet<>();
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    String s = getIslandDisplay(grid, i, j);
                    set.add(s);
                }
            }
        }
        return set.size();

    }

    private String getIslandDisplay(int[][] grid, int i, int j) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        StringBuilder builder = new StringBuilder();
        Queue<int[]> queue = new LinkedList<>();
        int m = grid.length;
        int n = grid[0].length;
        queue.offer(new int[] { i, j });
        builder.append(0).append(0);
        grid[i][j] = 0;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                    grid[nx][ny] = 0;
                    builder.append(nx - i).append(ny - j);
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        return builder.toString();

    }

    // 734. 句子相似性 (Sentence Similarity)
    public boolean areSentencesSimilar(String[] sentence1, String[] sentence2, List<List<String>> similarPairs) {
        int n1 = sentence1.length;
        int n2 = sentence2.length;
        if (n1 != n2) {
            return false;
        }
        Map<String, Set<String>> map = new HashMap<>();
        for (List<String> pair : similarPairs) {
            map.computeIfAbsent(pair.get(0), k -> new HashSet<>()).add(pair.get(1));
            map.computeIfAbsent(pair.get(1), k -> new HashSet<>()).add(pair.get(0));
        }
        for (int i = 0; i < n1; ++i) {
            if (sentence1[i].equals(sentence2[i])) {
                continue;
            }
            if (!map.getOrDefault(sentence1[i], new HashSet<>()).contains(sentence2[i])
                    && !map.getOrDefault(sentence2[i], new HashSet<>()).contains(sentence1[i])) {
                return false;
            }
        }
        return true;

    }

    // 737. 句子相似性 II (Sentence Similarity II) --plus
    public boolean areSentencesSimilarTwo(String[] sentence1, String[] sentence2, List<List<String>> similarPairs) {
        int n1 = sentence1.length;
        int n2 = sentence2.length;
        if (n1 != n2) {
            return false;
        }
        int count = 0;
        Map<String, Integer> map = new HashMap<>();
        for (String s : sentence1) {
            if (!map.containsKey(s)) {
                map.put(s, count++);
            }
        }
        for (String s : sentence2) {
            if (!map.containsKey(s)) {
                map.put(s, count++);
            }
        }
        for (List<String> pair : similarPairs) {
            for (String s : pair) {
                if (!map.containsKey(s)) {
                    map.put(s, count++);
                }
            }
        }
        Union737 union = new Union737(count);
        for (List<String> pair : similarPairs) {
            union.union(map.get(pair.get(0)), map.get(pair.get(1)));
        }
        for (int i = 0; i < n1; ++i) {
            if (!union.isConnected(map.get(sentence1[i]), map.get(sentence2[i]))) {
                return false;
            }
        }
        return true;

    }

    class Union737 {
        private int[] rank;
        private int[] parent;

        Union737(int n) {
            rank = new int[n];
            parent = new int[n];
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

    // 681. 最近时刻 (Next Closest Time) --plus
    public String nextClosestTime(String time) {
        boolean[] occur = new boolean[10];
        occur[time.charAt(0) - '0'] = true;
        occur[time.charAt(1) - '0'] = true;
        occur[time.charAt(3) - '0'] = true;
        occur[time.charAt(4) - '0'] = true;

        int original = Integer.parseInt(time.substring(0, 2)) * 60 + Integer.parseInt(time.substring(3));
        int cur = original + 1;
        while (cur < 24 * 60) {
            int hour = cur / 60;
            int min = cur % 60;
            if (occur[hour / 10] && occur[hour % 10] && occur[min / 10] && occur[min % 10]) {
                return String.format("%02d:%02d", hour, min);
            }
            ++cur;
        }
        cur = 0;
        while (cur <= original) {
            int hour = cur / 60;
            int min = cur % 60;
            if (occur[hour / 10] && occur[hour % 10] && occur[min / 10] && occur[min % 10]) {
                return String.format("%02d:%02d", hour, min);
            }
            ++cur;
        }
        return "";

    }

    // 742. 二叉树最近的叶节点 (Closest Leaf in a Binary Tree) --plus
    public int findClosestLeaf(TreeNode root, int k) {
        Queue<TreeNode> queue = new LinkedList<>();
        Map<Integer, List<Integer>> graph = new HashMap<>();
        Set<Integer> leaves = new HashSet<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                if (node.left == null && node.right == null) {
                    if (node.val == k) {
                        return k;
                    }
                    leaves.add(node.val);
                }
                if (node.left != null) {
                    graph.computeIfAbsent(node.val, o -> new ArrayList<>()).add(node.left.val);
                    graph.computeIfAbsent(node.left.val, o -> new ArrayList<>()).add(node.val);
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    graph.computeIfAbsent(node.val, o -> new ArrayList<>()).add(node.right.val);
                    graph.computeIfAbsent(node.right.val, o -> new ArrayList<>()).add(node.val);
                    queue.offer(node.right);
                }
            }
        }
        Set<Integer> visited = new HashSet<>();
        Queue<Integer> queue2 = new LinkedList<>();
        queue2.offer(k);
        visited.add(k);
        while (!queue2.isEmpty()) {
            int size = queue2.size();
            for (int i = 0; i < size; ++i) {
                int cur = queue2.poll();
                for (int neighbor : graph.getOrDefault(cur, new ArrayList<>())) {
                    if (visited.add(neighbor)) {
                        if (leaves.contains(neighbor)) {
                            return neighbor;
                        }
                        queue2.offer(neighbor);
                    }
                }
            }
        }
        return -1;
    }

    // 2450. Number of Distinct Binary Strings After Applying Operations --plus
    public int countDistinctStrings(String s, int k) {
        long res = 2l;
        final int mod = (int) (1e9 + 7);
        for (int x = 1; x <= s.length() - k; x++) {
            res = res * 2 % mod;
        }
        return (int) res;

    }

    // 1243. 数组变换 (Array Transformation) --plus
    public List<Integer> transformArray(int[] arr) {
        while (true) {
            boolean flag = false;
            int pre = arr[0];
            for (int i = 1; i < arr.length - 1; ++i) {
                if (pre < arr[i] && arr[i] > arr[i + 1]) {
                    flag = true;
                    pre = arr[i];
                    --arr[i];
                } else if (pre > arr[i] && arr[i] < arr[i + 1]) {
                    flag = true;
                    pre = arr[i];
                    ++arr[i];
                } else {
                    pre = arr[i];
                }
            }
            if (!flag) {
                return Arrays.stream(arr).boxed().collect(Collectors.toList());
            }
        }

    }

    // 1228. 等差数列中缺失的数字 (Missing Number In Arithmetic Progression) --plus
    public int missingNumber(int[] arr) {
        int n = arr.length;
        int d1 = arr[1] - arr[0];
        int d2 = arr[n - 1] - arr[n - 2];
        int d = 0;
        if (Math.abs(d1) < Math.abs(d2)) {
            d = d1;
        } else {
            d = d2;
        }
        for (int i = 1; i < n; ++i) {
            if (arr[i] - arr[i - 1] != d) {
                return arr[i - 1] + d;
            }
        }
        return arr[0];

    }

    // 1228. 等差数列中缺失的数字 (Missing Number In Arithmetic Progression) --plus
    public int missingNumber2(int[] arr) {
        int n = arr.length;
        int sum = (arr[0] + arr[n - 1]) * (n + 1) / 2;
        for (int a : arr) {
            sum -= a;
        }
        return sum;

    }

    // 325.和等于 k 的最长子数组长度 (Maximum Size Subarray Sum Equals k) --plus
    public int maxSubArrayLen(int[] nums, int k) {
        int res = 0;
        int n = nums.length;
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int prefix = 0;
        for (int i = 0; i < n; ++i) {
            prefix += nums[i];
            if (map.containsKey(prefix - k)) {
                res = Math.max(res, i - map.get(prefix - k));
            }
            map.putIfAbsent(prefix, i);
        }
        return res;

    }

    // 2422. Merge Operations to Turn Array Into a Palindrome
    public int minimumOperations2422(int[] nums) {
        int n = nums.length;
        int i = 0;
        int j = n - 1;
        long prefix = nums[0];
        long suffix = nums[n - 1];
        int res = 0;
        while (i < j) {
            if (prefix == suffix) {
                prefix = nums[++i];
                suffix = nums[--j];
            } else if (prefix < suffix) {
                ++res;
                prefix += nums[++i];
            } else {
                ++res;
                suffix += nums[--j];
            }
        }
        return res;

    }

    // 484. 寻找排列 (Find Permutation) --plus
    public int[] findPermutation(String s) {
        int n = s.length();
        int[] res = new int[n + 1];
        for (int i = 0; i < n + 1; ++i) {
            res[i] = i + 1;
        }
        int i = 0;
        while (i < n) {
            if (s.charAt(i) == 'D') {
                int j = i;
                while (j < n && s.charAt(j) == 'D') {
                    ++j;
                }
                swap484(res, i, j);
                i = j;
            } else {
                ++i;
            }
        }
        return res;

    }

    private void swap484(int[] arr, int i, int j) {
        while (i < j) {
            int temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
            ++i;
            --j;
        }
    }

    // 1055. 形成字符串的最短路径 (Shortest Way to Form String) --plus
    public int shortestWay(String source, String target) {
        Map<Character, Integer> map = new HashMap<>();
        for (int i = 0; i < source.length(); ++i) {
            map.putIfAbsent(source.charAt(i), i);
        }
        int n = target.length();
        int res = 0;
        int i = 0;
        while (i < n) {
            if (!map.containsKey(target.charAt(i))) {
                return -1;
            }
            ++res;
            int j = map.get(target.charAt(i));
            while (j < source.length() && i < target.length()) {
                if (source.charAt(j) == target.charAt(i)) {
                    ++i;
                }
                ++j;
            }
        }
        return res;
    }

    // 1150. 检查一个数是否在数组中占绝大多数 (Check If a Number Is Majority Element in a Sorted
    // Array) --plus
    public boolean isMajorityElement(int[] nums, int target) {
        int n = nums.length;
        int count = 0;
        for (int num : nums) {
            if (num == target) {
                ++count;
            } else if (num > target) {
                break;
            }
        }
        return count > n / 2;

    }

    // 1176. 健身计划评估 (Diet Plan Performance) --plus
    public int dietPlanPerformance(int[] calories, int k, int lower, int upper) {
        int n = calories.length;
        int res = 0;
        int sum = 0;
        for (int i = 0; i < k; ++i) {
            sum += calories[i];
        }
        if (sum < lower) {
            --res;
        } else if (sum > upper) {
            ++res;
        }

        for (int i = k; i < n; ++i) {
            sum += calories[i];
            sum -= calories[i - k];
            if (sum < lower) {
                --res;
            } else if (sum > upper) {
                ++res;
            }
        }
        return res;

    }

    // 790. 多米诺和托米诺平铺 (Domino and Tromino Tiling)
    public int numTilings(int n) {
        final int mod = (int) (1e9 + 7);
        int[][] dp = new int[n + 1][4];
        dp[0][3] = 1;
        for (int i = 1; i <= n; ++i) {
            dp[i][0] = dp[i - 1][3];
            dp[i][1] = (dp[i - 1][0] + dp[i - 1][2]) % mod;
            dp[i][2] = (dp[i - 1][0] + dp[i - 1][1]) % mod;
            dp[i][3] = ((dp[i - 1][0] % mod + dp[i - 1][1] % mod) % mod
                    + (dp[i - 1][2] % mod + dp[i - 1][3] % mod) % mod) % mod;
        }
        return dp[n][3];

    }

    // 790. 多米诺和托米诺平铺 (Domino and Tromino Tiling)
    private int[][] memo790;
    private int n790;

    public int numTilings2(int n) {
        this.n790 = n;
        this.memo790 = new int[n][3];
        return dfs790(0, 0);
    }

    private int dfs790(int i, int type) {
        if (i > n790) {
            return 0;
        }
        if (i == n790) {
            return type == 0 ? 1 : 0;
        }
        if (memo790[i][type] != 0) {
            return memo790[i][type];
        }
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        if (type == 0) {
            res = (res + dfs790(i + 1, 0)) % MOD;
            res = (res + dfs790(i + 2, 0)) % MOD;
            res = (res + dfs790(i + 2, 1)) % MOD;
            res = (res + dfs790(i + 2, 2)) % MOD;
        } else if (type == 1) {
            res = (res + dfs790(i + 1, 0)) % MOD;
            res = (res + dfs790(i + 1, 2)) % MOD;
        } else {
            res = (res + dfs790(i + 1, 0)) % MOD;
            res = (res + dfs790(i + 1, 1)) % MOD;
        }
        return memo790[i][type] = res;
    }

    // 1064. 不动点 (Fixed Point) --plus
    public int fixedPoint(int[] arr) {
        for (int i = 0; i < arr.length; ++i) {
            if (arr[i] == i) {
                return i;
            }
        }
        return -1;

    }

    // 1065. 字符串的索引对 (Index Pairs of a String) --plus
    public int[][] indexPairs(String text, String[] words) {
        List<int[]> res = new ArrayList<>();
        int n = text.length();
        for (int i = 0; i < n; ++i) {
            String sub = text.substring(i);
            for (String word : words) {
                if (sub.startsWith(word)) {
                    res.add(new int[] { i, i + word.length() - 1 });
                }
            }
        }
        Collections.sort(res, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return o1[1] - o2[1];
                }
                return o1[0] - o2[0];
            }

        });
        return res.toArray(new int[0][]);

    }

    // 6237. 不同的平均值数目 (Number of Distinct Averages)
    public int distinctAverages(int[] nums) {
        Arrays.sort(nums);
        Set<Integer> set = new HashSet<>();
        int i = 0;
        int j = nums.length - 1;
        while (i < j) {
            set.add(nums[i++] + nums[j--]);
        }
        return set.size();

    }

    // 6233. 温度转换 (Convert the Temperature)
    public double[] convertTemperature(double celsius) {
        return new double[] { celsius + 273.15d, celsius * 1.80d + 32.00d };
    }

    // 6234. 最小公倍数为 K 的子数组数目 (Number of Subarrays With LCM Equal to K)
    public int subarrayLCM(int[] nums, int k) {
        int res = 0;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (nums[i] > k) {
                continue;
            }
            int lcm = nums[i];
            for (int j = i; j < n; ++j) {
                if (nums[j] > k) {
                    break;
                }
                lcm = lcm6243(nums[j], lcm);
                if (lcm == k) {
                    ++res;
                } else if (lcm > k) {
                    break;
                }
            }
        }
        return res;

    }

    private int lcm6243(int a, int b) {
        return a * b / gcd6243(a, b);
    }

    private int gcd6243(int a, int b) {
        return b == 0 ? a : gcd6243(b, a % b);
    }

    // 6235. 逐层排序二叉树所需的最少操作数目 (Minimum Number of Operations to Sort a Binary Tree by
    // Level)
    public int minimumOperations(TreeNode root) {
        int res = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            int size = queue.size();
            int[] nums = new int[size];
            for (int i = 0; i < size; ++i) {
                TreeNode node = queue.poll();
                nums[i] = node.val;
                if (node.left != null) {
                    queue.offer(node.left);
                }
                if (node.right != null) {
                    queue.offer(node.right);
                }
            }
            res += getSwapCounts(nums);
        }
        return res;

    }

    private int getSwapCounts(int[] nums) {
        int res = 0;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            int min = nums[i];
            int minIndex = i;
            for (int j = i; j < n; ++j) {
                if (nums[j] < min) {
                    min = nums[j];
                    j = minIndex;
                }
            }
            if (minIndex != i) {
                swap6235(nums, i, minIndex);
                ++res;
            }
        }
        return res;
    }

    private void swap6235(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    // 1133. 最大唯一数 (Largest Unique Number) --plus
    public int largestUniqueNumber(int[] nums) {
        int max = 0;
        for (int num : nums) {
            max = Math.max(max, num);
        }
        int[] counts = new int[max + 1];
        for (int num : nums) {
            ++counts[num];
        }
        for (int i = max; i >= 0; --i) {
            if (counts[i] == 1) {
                return i;
            }
        }
        return -1;

    }

    // 1118. 一月有多少天 (Number of Days in a Month) --plus
    public int numberOfDays(int year, int month) {
        boolean isLeapYear = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        int[] monthOfDays = { 0, 31, isLeapYear ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        return monthOfDays[month];

    }

    // 1099. 小于 K 的两数之和 (Two Sum Less Than K) --plus
    public int twoSumLessThanK(int[] nums, int k) {
        int n = nums.length;
        int i = 0;
        int j = n - 1;
        Arrays.sort(nums);
        int res = -1;
        while (i < j) {
            int sum = nums[i] + nums[j];
            if (sum < k) {
                res = Math.max(res, sum);
                ++i;
            } else {
                --j;
            }
        }
        return res;

    }

    // 1885. 统计数对 (Count Pairs in Two Arrays) --plus
    public long countPairs(int[] nums1, int[] nums2) {
        int n = nums1.length;
        int[] diff = new int[n];
        for (int i = 0; i < n; ++i) {
            diff[i] = nums1[i] - nums2[i];
        }
        Arrays.sort(diff);
        long res = 0l;
        int left = 0;
        int right = n - 1;
        while (left < right) {
            if (diff[left] + diff[right] > 0) {
                res += right - left;
                --right;
            } else {
                ++left;
            }
        }
        return res;

    }

    // 2417. Closest Fair Integer --plus
    public int closestFair(int n) {
        while (true) {
            int digits = (int) Math.log10(n) + 1;
            if (digits % 2 == 1) {
                n = (int) Math.pow(10, digits);
            }
            int[] arr = new int[2];
            int num = n;
            while (num != 0) {
                int mod = num % 10;
                ++arr[mod % 2];
                num /= 10;
            }
            if (arr[0] == arr[1]) {
                return n;
            }
            ++n;
        }
    }

    // 1999. 最小的仅由两个数组成的倍数 (Smallest Greater Multiple Made of Two Digits)
    public int findInteger(int k, int digit1, int digit2) {
        if (digit1 > digit2) {
            return findInteger(k, digit2, digit1);
        }
        Queue<Integer> queue = new LinkedList<>();
        Set<Integer> set = new HashSet<>();
        queue.offer(0);
        set.add(0);
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                int num = queue.poll();
                if (num > Integer.MAX_VALUE / 10) {
                    continue;
                }
                int num1 = num * 10 + digit1;
                if (num1 > k && num1 % k == 0) {
                    return num1;
                }
                if (set.add(num1)) {
                    queue.offer(num1);
                }
                int num2 = num * 10 + digit2;
                if (num2 > k && num2 % k == 0) {
                    return num2;
                }
                if (set.add(num2)) {
                    queue.offer(num2);
                }
            }
        }
        return -1;

    }

    // 2021. 街上最亮的位置 (Brightest Position on Street) --plus
    public int brightestPosition(int[][] lights) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for (int[] light : lights) {
            map.put(light[0] - light[1], map.getOrDefault(light[0] - light[1], 0) + 1);
            map.put(light[0] + light[1] + 1, map.getOrDefault(light[0] + light[1] + 1, 0) - 1);
        }
        int res = 0;
        int max = 0;
        int cur = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            cur += entry.getValue();
            if (cur > max) {
                max = cur;
                res = entry.getKey();
            }
        }
        return res;
    }

    // 2015. 每段建筑物的平均高度 (Average Height of Buildings in Each Segment) --plus
    public int[][] averageHeightOfBuildings(int[][] buildings) {
        TreeMap<Integer, int[]> map = new TreeMap<>();
        for (int[] building : buildings) {
            if (!map.containsKey(building[0])) {
                map.put(building[0], new int[] { building[2], 1 });
            } else {
                int[] cur = map.get(building[0]);
                cur[0] += building[2];
                cur[1] += 1;
                map.put(building[0], cur);
            }
            if (!map.containsKey(building[1])) {
                map.put(building[1], new int[] { -building[2], -1 });
            } else {
                int[] cur = map.get(building[1]);
                cur[0] -= building[2];
                cur[1] -= 1;
                map.put(building[1], cur);
            }
        }
        int height = 0;
        int count = 0;
        List<int[]> list = new ArrayList<>();
        for (Map.Entry<Integer, int[]> entry : map.entrySet()) {
            height += entry.getValue()[0];
            count += entry.getValue()[1];
            list.add(new int[] { entry.getKey(), count == 0 ? 0 : height / count });
        }
        List<int[]> res = new ArrayList<>();
        int i = 0;
        while (i < list.size()) {
            if (list.get(i)[1] == 0) {
                ++i;
                continue;
            }
            int average = list.get(i)[1];
            int j = i + 1;
            while (j < list.size() && list.get(j)[1] == average) {
                ++j;
            }
            res.add(new int[] { list.get(i)[0], list.get(j)[0], average });
            i = j;
        }
        return res.toArray(new int[0][]);

    }

    // 1973. 值等于子节点值之和的节点数量 (Count Nodes Equal to Sum of Descendants) --plus
    private int res1973;

    public int equalToDescendants(TreeNode root) {
        dfs1973(root);
        return res1973;

    }

    private int dfs1973(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int sum = dfs1973(node.left) + dfs1973(node.right);
        if (node.val == sum) {
            ++res1973;
        }
        return sum + node.val;
    }

    // 2061. 扫地机器人清扫过的空间个数 (Number of Spaces Cleaning Robot Cleaned) --plus
    public int numberOfCleanRooms(int[][] room) {
        int m = room.length;
        int n = room[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                room[i][j] = -room[i][j];
            }
        }
        int i = 0;
        int j = 0;
        int[][] directions = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        int d = 0;
        int res = 0;
        while (true) {
            if (room[i][j] == 0) {
                ++res;
                room[i][j] |= 1 << d;
            } else if ((room[i][j] | (1 << d)) == room[i][j]) {
                return res;
            } else {
                room[i][j] |= 1 << d;
            }

            int nx = i + directions[d][0];
            int ny = j + directions[d][1];

            if (nx >= 0 && nx < m && ny >= 0 && ny < n && room[nx][ny] >= 0) {
                i = nx;
                j = ny;
            } else {
                d = (d + 1) % 4;
            }
        }

    }

    // 1564. 把箱子放进仓库里 I (Put Boxes Into the Warehouse I) --plus
    public int maxBoxesInWarehouse(int[] boxes, int[] warehouse) {
        int n = warehouse.length;
        int[] minHeight = new int[n];
        minHeight[0] = warehouse[0];
        for (int i = 1; i < n; ++i) {
            minHeight[i] = Math.min(minHeight[i - 1], warehouse[i]);
        }
        Arrays.sort(boxes);
        int i = 0;
        int j = n - 1;
        while (i < boxes.length && j >= 0) {
            if (boxes[i] <= minHeight[j]) {
                ++i;
            }
            --j;
        }
        return i;

    }

    // 1580. 把箱子放进仓库里 II (Put Boxes Into the Warehouse II) --plus
    public int maxBoxesInWarehouse1500(int[] boxes, int[] warehouse) {
        int n = warehouse.length;
        int[] minHeight = new int[n];
        minHeight[0] = warehouse[0];
        for (int i = 1; i < n; ++i) {
            minHeight[i] = Math.min(minHeight[i - 1], warehouse[i]);
        }
        int rightMin = Integer.MAX_VALUE;
        for (int i = n - 1; i >= 0; --i) {
            rightMin = Math.min(rightMin, warehouse[i]);
            minHeight[i] = Math.max(minHeight[i], rightMin);
        }

        Arrays.sort(boxes);
        Arrays.sort(minHeight);

        int i = 0;
        int j = 0;

        while (i < boxes.length && j < n) {
            if (boxes[i] <= minHeight[j]) {
                ++i;
            }
            ++j;
        }
        return i;

    }

    // 1102. 得分最高的路径 (Path With Maximum Minimum Value) --plus
    public int maximumMinimumPath(int[][] grid) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o2[2] - o1[2];
            }

        });
        queue.offer(new int[] { 0, 0, grid[0][0] });
        boolean[][] visited = new boolean[m][n];
        int res = Integer.MAX_VALUE;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            int val = cur[2];
            if (visited[x][y]) {
                continue;
            }
            visited[x][y] = true;
            res = Math.min(res, val);
            if (x == m - 1 && y == n - 1) {
                return res;
            }
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    queue.offer(new int[] { nx, ny, grid[nx][ny] });
                }
            }
        }
        return res;

    }

    // 1136. 并行课程 (Parallel Courses) --plus 拓扑排序
    public int minimumSemesters(int n, int[][] relations) {
        int[] degrees = new int[n];
        Map<Integer, List<Integer>> graph = new HashMap<>();
        for (int[] relation : relations) {
            graph.computeIfAbsent(relation[0] - 1, k -> new ArrayList<>()).add(relation[1] - 1);
            ++degrees[relation[1] - 1];
        }
        Queue<Integer> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (degrees[i] == 0) {
                queue.offer(i);
            }
        }
        int res = 0;
        while (!queue.isEmpty()) {
            ++res;
            int size = queue.size();
            for (int i = 0; i < size; ++i) {
                --n;
                int cur = queue.poll();
                for (int neighbor : graph.getOrDefault(cur, new ArrayList<>())) {
                    --degrees[neighbor];
                    if (degrees[neighbor] == 0) {
                        queue.offer(neighbor);
                    }
                }
            }
        }
        return n == 0 ? res : -1;

    }

    // 1135. 最低成本联通所有城市 (Connecting Cities With Minimum Cost) --plus 最小生成树
    public int minimumCost(int n, int[][] connections) {
        Arrays.sort(connections, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }

        });
        Union1135 union = new Union1135(n);
        int res = 0;
        int count = 0;
        for (int[] connection : connections) {
            if (count == n - 1) {
                return res;
            }
            if (union.isConnected(connection[0] - 1, connection[1] - 1)) {
                continue;
            }
            res += connection[2];
            union.union(connection[0] - 1, connection[1] - 1);
            ++count;
        }
        return count == n - 1 ? res : -1;
    }

    public class Union1135 {
        private int[] parent;
        private int[] rank;

        public Union1135(int n) {
            parent = new int[n];
            rank = new int[n];
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

    // 1236. 网络爬虫 (Web Crawler) --plus
    /**
     * // This is the HtmlParser's API interface. // You should not implement it, or
     * speculate about its implementation
     */
    interface HtmlParser {
        public List<String> getUrls(String url);
    }

    public List<String> crawl(String startUrl, HtmlParser htmlParser) {
        Set<String> set = new HashSet<>();
        Queue<String> queue = new LinkedList<>();
        String prefix = getHostName1236(startUrl);
        set.add(startUrl);
        queue.offer(startUrl);
        while (!queue.isEmpty()) {
            String url = queue.poll();
            for (String neighbor : htmlParser.getUrls(url)) {
                String pre = getHostName1236(neighbor);
                if (pre.equals(prefix) && set.add(neighbor)) {
                    queue.offer(neighbor);
                }
            }
        }
        return new ArrayList<>(set);

    }

    private String getHostName1236(String s) {
        String url = s.substring(7);
        int index = url.indexOf("/");
        if (index == -1) {
            return s;
        }
        return s.substring(0, index + 7);
    }

    // 1257. 最小公共区域 (Smallest Common Region) --plus
    public String findSmallestRegion(List<List<String>> regions, String region1, String region2) {
        Map<String, String> map = new HashMap<>();
        for (List<String> region : regions) {
            String parent = region.get(0);
            for (int i = 1; i < region.size(); ++i) {
                map.put(region.get(i), parent);
            }
        }
        Set<String> parents = new HashSet<>();

        do {
            parents.add(region1);
            region1 = map.get(region1);
        } while (region1 != null);

        do {
            if (parents.contains(region2)) {
                return region2;
            }
            parents.add(region2);
            region2 = map.get(region2);
        } while (region2 != null);

        return "";

    }

    // 1257. 最小公共区域 (Smallest Common Region) --plus
    public String findSmallestRegion2(List<List<String>> regions, String region1, String region2) {
        Map<String, String> map = new HashMap<>();
        for (List<String> region : regions) {
            String parent = region.get(0);
            for (int i = 1; i < region.size(); ++i) {
                map.put(region.get(i), parent);
            }
        }
        String r1 = region1;
        String r2 = region2;
        while (!r1.equals(r2)) {
            r1 = map.getOrDefault(r1, region2);
            r2 = map.getOrDefault(r2, region1);
        }
        return r1;

    }

    // 2107. 分享 K 个糖果后独特口味的数量 (Number of Unique Flavors After Sharing K Candies)
    // --plus
    public int shareCandies(int[] candies, int k) {
        int max = 0;
        for (int candy : candies) {
            max = Math.max(max, candy);
        }
        int[] counts = new int[max + 1];
        for (int candy : candies) {
            ++counts[candy];
        }
        for (int i = 0; i < k; ++i) {
            --counts[candies[i]];
        }
        int left = 0;
        for (int count : counts) {
            if (count > 0) {
                ++left;
            }
        }
        int res = left;
        for (int i = k; i < candies.length; ++i) {
            if (--counts[candies[i]] == 0) {
                --left;
            }
            if (++counts[candies[i - k]] == 1) {
                ++left;
            }
            res = Math.max(res, left);
        }
        return res;

    }

    // 1740. 找到二叉树中的距离 (Find Distance in a Binary Tree) --plus
    public int findDistance(TreeNode root, int p, int q) {
        if (p == q) {
            return 0;
        }
        Map<Integer, List<Integer>> graph = new HashMap<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.offer(root);
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node.left != null) {
                graph.computeIfAbsent(node.val, k -> new ArrayList<>()).add(node.left.val);
                graph.computeIfAbsent(node.left.val, k -> new ArrayList<>()).add(node.val);
                queue.offer(node.left);
            }
            if (node.right != null) {
                graph.computeIfAbsent(node.val, k -> new ArrayList<>()).add(node.right.val);
                graph.computeIfAbsent(node.right.val, k -> new ArrayList<>()).add(node.val);
                queue.offer(node.right);
            }
        }
        int res = 0;
        Set<Integer> visited = new HashSet<>();
        visited.add(p);
        Queue<Integer> queue2 = new LinkedList<>();
        queue2.offer(p);
        while (!queue2.isEmpty()) {
            int size = queue2.size();
            ++res;
            for (int i = 0; i < size; ++i) {
                int cur = queue2.poll();
                for (int neighbor : graph.getOrDefault(cur, new ArrayList<>())) {
                    if (neighbor == q) {
                        return res;
                    }
                    if (visited.add(neighbor)) {
                        queue2.offer(neighbor);
                    }
                }
            }
        }
        return -1;

    }

    // 1258. 近义词句子 (Synonymous Sentences) --plus
    public List<String> generateSentences(List<List<String>> synonyms, String text) {
        Map<String, Integer> map = new HashMap<>();
        int count = 0;
        for (List<String> synouym : synonyms) {
            for (String s : synouym) {
                if (!map.containsKey(s)) {
                    map.put(s, count++);
                }
            }
        }
        String[] split = text.split("\\s+");

        List<String> res = new ArrayList<>();
        Union1258 union = new Union1258(count);
        for (List<String> synonym : synonyms) {
            union.union(map.get(synonym.get(0)), map.get(synonym.get(1)));
        }
        dfs1258(res, map, union, split, 0);
        Collections.sort(res);
        return res;

    }

    private void dfs1258(List<String> res, Map<String, Integer> map, Union1258 union, String[] split, int i) {
        if (i == split.length) {
            res.add(String.join(" ", split));
            return;
        }
        if (map.containsKey(split[i])) {
            String originalString = split[i];
            for (Map.Entry<String, Integer> entry : map.entrySet()) {
                if (union.isConnected(entry.getValue(), map.get(originalString))) {
                    split[i] = entry.getKey();
                    dfs1258(res, map, union, split.clone(), i + 1);
                    split[i] = originalString;
                }
            }
        } else {
            dfs1258(res, map, union, split.clone(), i + 1);
        }
    }

    public class Union1258 {
        private int[] parent;
        private int[] rank;

        public Union1258(int n) {
            parent = new int[n];
            rank = new int[n];
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

    // 253. 会议室 II (Meeting Rooms II) --plus
    public int minMeetingRooms(int[][] intervals) {
        int max = 0;
        for (int[] interval : intervals) {
            max = Math.max(max, interval[1]);
        }
        int[] diff = new int[max + 1];
        for (int[] interval : intervals) {
            ++diff[interval[0]];
            --diff[interval[1]];
        }
        int res = diff[0];
        for (int i = 1; i < diff.length; ++i) {
            diff[i] += diff[i - 1];
            res = Math.max(res, diff[i]);
        }
        return res;

    }

    // 891. 子序列宽度之和 (Sum of Subsequence Widths)
    public int sumSubseqWidths(int[] nums) {
        int n = nums.length;
        Arrays.sort(nums);
        long x = nums[0];
        long y = 2l;
        long res = 0l;
        final int mod = (int) (1e9 + 7);
        for (int j = 1; j < n; ++j) {
            res = (res + nums[j] * (y - 1) - x) % mod;
            x = (x * 2 + nums[j]) % mod;
            y = y * 2 % mod;
        }
        return (int) (res % mod);

    }

    // 2475. 数组中不等三元组的数目 (Number of Unequal Triplets in Array)
    public int unequalTriplets(int[] nums) {
        int n = nums.length;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                for (int k = j + 1; k < n; ++k) {
                    if (nums[i] != nums[j] && nums[j] != nums[k] && nums[i] != nums[k]) {
                        ++res;
                    }
                }
            }
        }
        return res;

    }

    // 2475. 数组中不等三元组的数目 (Number of Unequal Triplets in Array)
    public int unequalTriplets2(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int left = 0;
        int right = 1;
        int res = 0;
        while (right < n) {
            if (nums[left] != nums[right]) {
                res += left * (right - left) * (n - right);
                left = right;
            } else {
                ++right;
            }
        }
        return res;

    }

    // 2475. 数组中不等三元组的数目 (Number of Unequal Triplets in Array)
    public int unequalTriplets3(int[] nums) {
        int res = 0;
        int[] cnts = new int[1001];
        for (int num : nums) {
            ++cnts[num];
        }
        int n = nums.length;
        int a = 0;
        int c = n;
        for (int cnt : cnts) {
            c -= cnt;
            res += a * cnt * c;
            a += cnt;
        }
        return res;

    }

    // 6242. 二叉搜索树最近节点查询
    public List<List<Integer>> closestNodes(TreeNode root, List<Integer> queries) {
        TreeSet<Integer> set = new TreeSet<>();
        dfs6242(set, root);

        List<List<Integer>> res = new ArrayList<>();
        for (int query : queries) {
            Integer floor = set.floor(query);
            if (floor == null) {
                floor = -1;
            }

            Integer ceil = set.ceiling(query);
            if (ceil == null) {
                ceil = -1;
            }
            res.add(List.of(floor, ceil));

        }
        return res;
    }

    private void dfs6242(TreeSet<Integer> list, TreeNode root) {
        if (root == null) {
            return;
        }
        dfs6242(list, root.left);
        list.add(root.val);
        dfs6242(list, root.right);

    }

    // 2477. 到达首都的最少油耗 (Minimum Fuel Cost to Report to the Capital)
    public long minimumFuelCost(int[][] roads, int seats) {
        long res = 0;
        int n = roads.length + 1;
        List<Integer>[] g = new ArrayList[n];
        Arrays.setAll(g, k -> new ArrayList<>());
        int[] deg = new int[n];
        long[] cnt = new long[n];
        Arrays.fill(cnt, 1L);
        for (int[] r : roads) {
            int u = r[0];
            int v = r[1];
            g[u].add(v);
            g[v].add(u);
            ++deg[u];
            ++deg[v];
        }
        Queue<Integer> q = new LinkedList<>();
        for (int i = 1; i < n; ++i) {
            if (deg[i] == 1) {
                q.offer(i);
            }
        }
        while (!q.isEmpty()) {
            int x = q.poll();
            if (x == 0) {
                continue;
            }
            --deg[x];
            for (int y : g[x]) {
                if (deg[y] == 0) {
                    continue;
                }
                --deg[y];
                res += (cnt[x] + seats - 1) / seats;
                cnt[y] += cnt[x];
                if (deg[y] == 1) {
                    q.offer(y);
                }
            }
        }
        return res;

    }

    // 1348. 推文计数 (Tweet Counts Per Frequency)
    class TweetCounts {
        private Map<String, List<Integer>> map;

        public TweetCounts() {
            map = new HashMap<>();

        }

        public void recordTweet(String tweetName, int time) {
            map.computeIfAbsent(tweetName, k -> new ArrayList<>()).add(Integer.MIN_VALUE);
            map.computeIfAbsent(tweetName, k -> new ArrayList<>()).add(Integer.MAX_VALUE);
            map.computeIfAbsent(tweetName, k -> new ArrayList<>()).add(time);
        }

        public List<Integer> getTweetCountsPerFrequency(String freq, String tweetName, int startTime, int endTime) {
            List<Integer> res = new ArrayList<>();

            List<Integer> list = map.getOrDefault(tweetName, new ArrayList<>());
            Collections.sort(list);
            if (list.isEmpty()) {
                return res;
            }
            int divider = 0;
            if ("minute".equals(freq)) {
                divider = 60;
            } else if ("hour".equals(freq)) {
                divider = 3600;
            } else if ("day".equals(freq)) {
                divider = 86400;
            }
            while (startTime <= endTime) {
                int min = startTime;
                int max = Math.min(startTime + divider - 1, endTime);
                int leftIndex = binarySearchCeiling1348(list, min);
                int rightIndex = binarySearchFloor1348(list, max);
                res.add(rightIndex - leftIndex + 1);
                startTime = max + 1;
            }
            return res;

        }

        private int binarySearchFloor1348(List<Integer> list, int target) {
            int left = 0;
            int right = list.size();
            int res = -1;
            while (left <= right) {
                int mid = left + ((right - left) >>> 1);
                if (list.get(mid) <= target) {
                    res = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return res;

        }

        private int binarySearchCeiling1348(List<Integer> list, int target) {
            int left = 0;
            int right = list.size() - 1;
            int res = -1;
            while (left <= right) {
                int mid = left + ((right - left) >>> 1);
                if (list.get(mid) >= target) {
                    res = mid;
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            return res;
        }
    }

    // 1191. K 次串联后最大子数组之和 (K-Concatenation Maximum Sum)
    public int kConcatenationMaxSum(int[] arr, int k) {
        final int mod = (int) (1e9 + 7);
        int n = arr.length;
        long preSum = 0l;
        long sufSum = 0l;
        long maxSum = 0l;
        long minPreSum = 0l;
        long maxPreSum = 0l;
        long maxSufSum = 0l;
        for (int num : arr) {
            preSum += num;
            maxSum = Math.max(maxSum, preSum - minPreSum);
            minPreSum = Math.min(minPreSum, preSum);
            maxPreSum = Math.max(maxPreSum, preSum);
        }

        for (int i = n - 1; i >= 0; --i) {
            sufSum += arr[i];
            maxSufSum = Math.max(maxSufSum, sufSum);
        }
        long res = 0l;
        res = Math.max(res, maxSum);
        if (k >= 2) {
            res = Math.max(res, Math.max(preSum * (k - 2) + maxPreSum + maxSufSum, maxPreSum + maxSufSum));
        }
        return (int) (res % mod);

    }

    // 878. 第 N 个神奇数字 (Nth Magical Number)
    public int nthMagicalNumber(int n, int a, int b) {
        final int mod = (int) (1e9 + 7);
        long left = Math.min(a, b);
        long right = (long) n * Math.min(a, b);
        int c = lcm878(a, b);
        long res = -1;
        while (left <= right) {
            long mid = left + ((right - left) >>> 1);
            long num = mid / a + mid / b - mid / c;
            if (num >= n) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return (int) (res % mod);

    }

    private int lcm878(int a, int b) {
        return a * b / gcd878(a, b);
    }

    private int gcd878(int a, int b) {
        return b == 0 ? a : gcd878(b, a % b);
    }

    // 2400. 恰好移动 k 步到达某一位置的方法数目 (Number of Ways to Reach a Position After Exactly k
    // Steps)
    private int endPos2400;
    private int[][] memo2400;

    public int numberOfWays(int startPos, int endPos, int k) {
        this.endPos2400 = endPos + 1000;
        startPos += 1000;
        memo2400 = new int[Math.max(startPos, this.endPos2400) + k / 2 + 1][k + 1];
        for (int i = 0; i < Math.max(startPos, this.endPos2400) + k / 2 + 1; ++i) {
            Arrays.fill(memo2400[i], -1);
        }
        return dfs2400(startPos, k);

    }

    private int dfs2400(int startPos, int k) {
        if (Math.abs(startPos - endPos2400) > k) {
            return 0;
        }
        if (k == 0) {
            return startPos == endPos2400 ? 1 : 0;
        }
        if (memo2400[startPos][k] != -1) {
            return memo2400[startPos][k];
        }
        final int MOD = (int) (1e9 + 7);
        int res = 0;
        res = (res + dfs2400(startPos - 1, k - 1)) % MOD;
        res = (res + dfs2400(startPos + 1, k - 1)) % MOD;
        return memo2400[startPos][k] = res;
    }

    // 1540. K 次操作转变字符串 (Can Convert String in K Moves)
    public boolean canConvertString(String s, String t, int k) {
        int m = s.length();
        int n = t.length();
        if (m != n) {
            return false;
        }
        int[] counts = new int[26];
        int div = k / 26;
        int mod = k % 26;
        for (int i = 1; i < 26; ++i) {
            counts[i] = div + (i <= mod ? 1 : 0);
        }
        for (int i = 0; i < m; ++i) {
            char c1 = s.charAt(i);
            char c2 = t.charAt(i);
            int diff = (c2 - c1 + 26) % 26;
            if (diff > 0 && --counts[diff] < 0) {
                return false;
            }
        }
        return true;
    }

    // 1093. 大样本统计 (Statistics from a Large Sample)
    public double[] sampleStats(int[] count) {
        int min = -1;
        int max = -1;
        long sum = 0;
        int mojority = -1;
        int mojorityCount = 0;
        for (int i = 0; i < count.length; ++i) {
            if (count[i] != 0) {
                if (min == -1) {
                    min = i;
                }
                max = i;
            }
            if (count[i] > mojorityCount) {
                mojorityCount = count[i];
                mojority = i;
            }
            sum += (long) count[i] * i;
            if (i > 0) {
                count[i] += count[i - 1];
            }
        }
        double median = binarySearch1093(count, count[255] / 2 + 1);
        if (count[255] % 2 == 0) {
            median = (binarySearch1093(count, count[255] / 2) + median) / 2d;
        }
        return new double[] { min, max, (double) sum / count[255], median, mojority };

    }

    private double binarySearch1093(int[] count, int target) {
        int left = 0;
        int right = count.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (count[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }

    // 1396. 设计地铁系统 (Design Underground System)
    class UndergroundSystem {
        private Map<Integer, Bean> map;
        private Map<String, long[]> countMap;

        class Bean {
            String startStation;
            int startTime;

            Bean(String startStation, int startTime) {
                this.startStation = startStation;
                this.startTime = startTime;
            }
        }

        public UndergroundSystem() {
            map = new HashMap<>();
            countMap = new HashMap<>();

        }

        public void checkIn(int id, String stationName, int t) {
            map.put(id, new Bean(stationName, t));
        }

        public void checkOut(int id, String stationName, int t) {
            Bean bean = map.get(id);
            long[] sum = countMap.getOrDefault(bean.startStation + "_" + stationName, new long[] { 0l, 0l });
            sum[0] += (long) t - bean.startTime;
            sum[1] += 1;
            countMap.put(bean.startStation + "_" + stationName, sum);
            map.remove(id);
        }

        public double getAverageTime(String startStation, String endStation) {
            long[] sum = countMap.get(startStation + "_" + endStation);
            return (double) sum[0] / sum[1];

        }
    }

    // 855. 考场就座 (Exam Room)
    class ExamRoom {
        private TreeSet<Integer> set;
        private int n;

        public ExamRoom(int n) {
            this.n = n;
            this.set = new TreeSet<>();
        }

        public int seat() {
            if (set.isEmpty()) {
                set.add(0);
                return 0;
            }
            int d = 0;
            int res = -1;
            int x = set.first();
            int y = set.last();
            if (x >= n - 1 - y) {
                d = x;
                res = 0;
            } else {
                d = n - 1 - y;
                res = n - 1;
            }
            int pre = -1;
            for (int cur : set) {
                if (pre == -1) {
                    pre = cur;
                } else if ((cur - pre) / 2 > d || (cur - pre) / 2 == d && (cur + pre) / 2 < res) {
                    d = (cur - pre) / 2;
                    res = (cur + pre) / 2;
                }
                pre = cur;
            }
            set.add(res);
            return res;
        }

        public void leave(int p) {
            set.remove(p);
        }
    }

    // 1432. 改变一个整数能得到的最大差值 (Max Difference You Can Get From Changing an Integer)
    public int maxDiff(int num) {
        char[] a = String.valueOf(num).toCharArray();
        return check1432(num, '9') - (a[0] != '1' ? check1432(num, '1') : check1432(num, '1', '0'));

    }

    private int check1432(int num, char x, char y) {
        char[] a = String.valueOf(num).toCharArray();
        for (int i = 0; i < a.length; ++i) {
            if (a[i] != x && a[i] != y) {
                char t = a[i];
                for (int j = i; j < a.length; ++j) {
                    if (a[j] == t) {
                        a[j] = '0';
                    }
                }
                break;
            }
        }
        return Integer.parseInt(String.valueOf(a));

    }

    private int check1432(int num, char t) {
        char[] a = String.valueOf(num).toCharArray();
        for (int i = 0; i < a.length; ++i) {
            if (a[i] != t) {
                char c = a[i];
                for (int j = i; j < a.length; ++j) {
                    if (a[j] == c) {
                        a[j] = t;
                    }
                }
                break;
            }
        }
        return Integer.parseInt(String.valueOf(a));
    }

    // 882. 细分图中的可到达节点 (Reachable Nodes In Subdivided Graph)
    public int reachableNodes(int[][] edges, int maxMoves, int n) {
        Map<Integer, List<int[]>> graph = new HashMap<>();
        for (int[] edge : edges) {
            graph.computeIfAbsent(edge[0], k -> new ArrayList<>()).add(new int[] { edge[1], edge[2] + 1 });
            graph.computeIfAbsent(edge[1], k -> new ArrayList<>()).add(new int[] { edge[0], edge[2] + 1 });
        }
        int[] dis = dijkstra882(graph, n);
        int res = 0;
        for (int d : dis) {
            if (d <= maxMoves) {
                ++res;
            }
        }
        for (int[] edge : edges) {
            int node1 = edge[0];
            int node2 = edge[1];
            int d = edge[2];
            int a = Math.max(maxMoves - dis[node1], 0);
            int b = Math.max(maxMoves - dis[node2], 0);
            res += Math.min(a + b, d);
        }
        return res;

    }

    private int[] dijkstra882(Map<Integer, List<int[]>> graph, int n) {
        int[] dis = new int[n];
        Arrays.fill(dis, Integer.MAX_VALUE);
        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[1] - o2[1];
            }

        });

        dis[0] = 0;
        queue.offer(new int[] { 0, 0 });
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int node = cur[0];
            int d = cur[1];
            if (d > dis[node]) {
                continue;
            }
            for (int[] neighbor : graph.getOrDefault(node, new ArrayList<>())) {
                int neighborNode = neighbor[0];
                int neighborDis = neighbor[1] + d;
                if (neighborDis < dis[neighborNode]) {
                    dis[neighborNode] = neighborDis;
                    queue.offer(new int[] { neighborNode, neighborDis });
                }
            }
        }
        return dis;
    }

    // 1921. 消灭怪物的最大数量 (Eliminate Maximum Number of Monsters)
    public int eliminateMaximum(int[] dist, int[] speed) {
        int n = dist.length;
        Queue<Integer> queue = new PriorityQueue<>();
        for (int i = 0; i < n; ++i) {
            queue.offer((dist[i] - 1) / speed[i] + 1);
        }
        int res = 0;
        while (!queue.isEmpty()) {
            if (res < queue.poll()) {
                ++res;
            } else {
                break;
            }
        }
        return res;

    }

    // 6249. 分割圆的最少切割次数 (Minimum Cuts to Divide a Circle)
    public int numberOfCuts(int n) {
        if (n == 1) {
            return 0;
        }
        if (n % 2 == 0) {
            return n / 2;
        }
        return n;
    }

    // 6277. 行和列中一和零的差值 (Difference Between Ones and Zeros in Row and Column)
    public int[][] onesMinusZeros(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[] row1 = new int[m];
        int[] col1 = new int[n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                row1[i] += grid[i][j];
                col1[j] += grid[i][j];
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                grid[i][j] = row1[i] + col1[j] - (m - row1[i]) - (n - col1[j]);
            }
        }
        return grid;

    }

    // 6250. 商店的最少代价 (Minimum Penalty for a Shop)
    public int bestClosingTime(String customers) {
        int curCost = 0;
        int minCost = 0;
        int res = customers.length();
        for (char c : customers.toCharArray()) {
            if (c == 'N') {
                ++curCost;
            }
        }
        minCost = curCost;
        for (int i = customers.length() - 1; i >= 0; --i) {
            if (customers.charAt(i) == 'Y') {
                ++curCost;
            } else {
                --curCost;
            }
            if (curCost <= minCost) {
                minCost = curCost;
                res = i;
            }
        }
        return res;

    }

    // 1904. 你完成的完整对局数 (The Number of Full Rounds You Have Played)
    public int numberOfRounds(String loginTime, String logoutTime) {
        int minute1 = getMinutes1904(loginTime);
        int minute2 = getMinutes1904(logoutTime);
        if (minute1 > minute2) {
            minute2 += 24 * 60;
        }
        int mod = minute1 % 15;
        minute1 += (15 - mod) % 15;
        minute2 = minute2 / 15 * 15;
        return Math.max(0, (minute2 - minute1) / 15);
    }

    private int getMinutes1904(String time) {
        int h = Integer.parseInt(time.substring(0, 2));
        int m = Integer.parseInt(time.substring(3));
        return h * 60 + m;
    }

    // 2485. 找出中枢整数 (Find the Pivot Integer)
    public int pivotInteger(int n) {
        for (int i = 1; i <= n; ++i) {
            if ((1 + i) * i == (i + n) * (n - i + 1)) {
                return i;
            }
        }
        return -1;

    }

    // 2486. 追加字符以获得子序列 (Append Characters to String to Make Subsequence)
    public int appendCharacters(String s, String t) {
        int m = s.length();
        int n = t.length();
        int i = 0;
        int j = 0;
        while (i < m && j < n) {
            if (s.charAt(i) == t.charAt(j)) {
                ++j;
            }
            ++i;
        }
        return n - j;

    }

    // 2487. 从链表中移除节点 (Remove Nodes From Linked List)
    public ListNode removeNodes(ListNode head) {
        ListNode dummy = new ListNode(Integer.MAX_VALUE);
        Stack<ListNode> stack = new Stack<>();
        stack.push(dummy);
        ListNode pre = dummy;
        while (head != null) {
            while (stack.peek().val < head.val) {
                stack.pop();
                pre = stack.peek();
            }
            stack.push(head);
            pre.next = head;
            pre = head;
            head = head.next;
        }
        return dummy.next;

    }

    // 2487. 从链表中移除节点 (Remove Nodes From Linked List)
    public ListNode removeNodes2(ListNode head) {
        ListNode res = reverse2487(head);
        ListNode cur = res;
        while (cur != null) {
            ListNode next = cur.next;
            while (next != null && cur.val > next.val) {
                next = next.next;
            }
            cur.next = next;
            cur = next;
        }
        res = reverse2487(res);
        return res;

    }

    private ListNode reverse2487(ListNode head) {
        ListNode pre = null;
        while (head != null) {
            ListNode next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }

    // 1177. 构建回文串检测 (Can Make Palindrome from Substring)
    public List<Boolean> canMakePaliQueries(String s, int[][] queries) {
        int[] pre = new int[s.length() + 1];
        for (int i = 0; i < s.length(); ++i) {
            pre[i + 1] = pre[i] ^ (1 << (s.charAt(i) - 'a'));
        }
        List<Boolean> res = new ArrayList<>();
        for (int[] q : queries) {
            int xor = pre[q[1] + 1] ^ pre[q[0]];
            res.add(Integer.bitCount(xor) - q[2] * 2 <= 1);
        }
        return res;
    }

    // 1818. 绝对差值和 (Minimum Absolute Sum Difference)
    public int minAbsoluteSumDiff(int[] nums1, int[] nums2) {
        long sum = 0l;
        int n = nums1.length;
        TreeSet<Integer> set = new TreeSet<>();
        for (int i = 0; i < n; ++i) {
            sum += Math.abs(nums1[i] - nums2[i]);
            set.add(nums1[i]);
        }
        long res = sum;
        for (int i = 0; i < n; ++i) {
            int original = Math.abs(nums1[i] - nums2[i]);
            int minus = Integer.MAX_VALUE;
            Integer floor = set.floor(nums2[i]);
            Integer ceiling = set.ceiling(nums2[i]);
            if (floor != null) {
                minus = Math.min(minus, Math.abs(nums2[i] - floor));
            }
            if (ceiling != null) {
                minus = Math.min(minus, Math.abs(nums2[i] - ceiling));
            }
            res = Math.min(res, sum - original + minus);
        }
        final int mod = (int) (1e9 + 7);
        return (int) (res % mod);

    }

    // 1818. 绝对差值和 (Minimum Absolute Sum Difference)
    public int minAbsoluteSumDiff2(int[] nums1, int[] nums2) {
        long sum = 0l;
        int n = nums1.length;
        int[] sort1 = new int[n];
        for (int i = 0; i < n; ++i) {
            sum += Math.abs(nums1[i] - nums2[i]);
            sort1[i] = nums1[i];
        }
        Arrays.sort(sort1);
        long res = sum;
        for (int i = 0; i < n; ++i) {
            int original = Math.abs(nums1[i] - nums2[i]);
            int minus = Integer.MAX_VALUE;
            int floor = binarySearchFloor1818(sort1, nums2[i]);
            int ceiling = binarySearchCeiling1818(sort1, nums2[i]);
            if (floor != -1) {
                minus = Math.min(minus, Math.abs(floor - nums2[i]));
            }
            if (ceiling != -1) {
                minus = Math.min(minus, Math.abs(ceiling - nums2[i]));
            }
            res = Math.min(res, sum - original + minus);
        }
        final int mod = (int) (1e9 + 7);
        return (int) (res % mod);

    }

    // 找排序数组nums中 <= target 的最大元素
    private int binarySearchFloor1818(int[] nums, int target) {
        int n = nums.length;
        if (target < nums[0]) {
            return -1;
        }
        if (target >= nums[n - 1]) {
            return nums[n - 1];
        }
        int left = 0;
        int right = n - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] <= target) {
                res = nums[mid];
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;
    }

    // 找排序数组nums中 >= target 的最小元素
    private int binarySearchCeiling1818(int[] nums, int target) {
        int n = nums.length;
        if (nums[0] >= target) {
            return nums[0];
        }
        if (nums[n - 1] < target) {
            return -1;
        }
        int left = 0;
        int right = n - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] >= target) {
                res = nums[mid];
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    // 1577. 数的平方等于两数乘积的方法数 (Number of Ways Where Square of Number Is Equal to
    // Product of Two Numbers)
    public int numTriplets(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map1 = new HashMap<>();
        Map<Integer, Integer> map2 = new HashMap<>();
        for (int num : nums1) {
            map1.put(num, map1.getOrDefault(num, 0) + 1);
        }
        for (int num : nums2) {
            map2.put(num, map2.getOrDefault(num, 0) + 1);
        }
        return getTriplets1577(map1, map2) + getTriplets1577(map2, map1);

    }

    private int getTriplets1577(Map<Integer, Integer> map1, Map<Integer, Integer> map2) {
        int res = 0;
        Set<Integer> set1 = map1.keySet();
        Set<Integer> set2 = map2.keySet();
        for (int num1 : set1) {
            int count1 = map1.get(num1);
            long square = (long) num1 * num1;
            for (int num2 : set2) {
                if (square % num2 == 0 && square / num2 <= Integer.MAX_VALUE) {
                    int count2 = map2.get(num2);
                    int num3 = (int) (square / num2);

                    if (num2 == num3) {
                        res += count1 * count2 * (count2 - 1) / 2;
                    } else if (num2 < num3 && set2.contains(num3)) {
                        int count3 = map2.get(num3);
                        res += count1 * count2 * count3;
                    }
                }
            }
        }
        return res;
    }

    // 866. 回文素数 (Prime Palindrome)
    public int primePalindrome(int n) {
        for (int L = 1; L <= 5; ++L) {
            for (int root = (int) Math.pow(10, L - 1); root < Math.pow(10, L); ++root) {
                StringBuilder builder = new StringBuilder(String.valueOf(root));
                for (int k = L - 2; k >= 0; --k) {
                    builder.append(builder.charAt(k));
                }
                int num = Integer.parseInt(builder.toString());
                if (num >= n && isPrime866(num)) {
                    return num;
                }
            }

            for (int root = (int) Math.pow(10, L - 1); root < Math.pow(10, L); ++root) {
                StringBuilder builder = new StringBuilder(String.valueOf(root));
                for (int k = L - 1; k >= 0; --k) {
                    builder.append(builder.charAt(k));
                }
                int num = Integer.parseInt(builder.toString());
                if (num >= n && isPrime866(num)) {
                    return num;
                }
            }
        }
        return -1;

    }

    private boolean isPrime866(int num) {
        if (num < 2) {
            return false;
        }
        int sqrt = (int) Math.sqrt(num);
        for (int i = 2; i <= sqrt; ++i) {
            if (num % i == 0) {
                return false;
            }
        }
        return true;
    }

    // 1339. 分裂二叉树的最大乘积 (Maximum Product of Splitted Binary Tree)
    private int sum1339;
    private int best1339;

    public int maxProduct(TreeNode root) {
        dfs_sum1339(root);
        dfs_best1339(root);
        final int mod = (int) (1e9 + 7);
        return (int) ((long) best1339 * (sum1339 - best1339) % mod);
    }

    private void dfs_sum1339(TreeNode node) {
        if (node == null) {
            return;
        }
        sum1339 += node.val;
        dfs_sum1339(node.left);
        dfs_sum1339(node.right);
    }

    private int dfs_best1339(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int cur = dfs_best1339(node.left) + dfs_best1339(node.right) + node.val;
        if (Math.abs(cur * 2 - sum1339) < Math.abs(best1339 * 2 - sum1339)) {
            best1339 = cur;
        }
        return cur;
    }

    // 276. 栅栏涂色 (Paint Fence) --plus
    public int numWays(int n, int k) {
        if (n <= 1) {
            return k;
        }
        int[] dp = new int[n];
        dp[0] = k;
        dp[1] = k * k;
        for (int i = 2; i < n; ++i) {
            dp[i] = dp[i - 1] * (k - 1) + dp[i - 2] * (k - 1);
        }
        return dp[n - 1];

    }

    // 276. 栅栏涂色 (Paint Fence) --plus
    public int numWays2(int n, int k) {
        if (n <= 1) {
            return k;
        }
        int a = k;
        int b = k * k;
        for (int i = 2; i < n; ++i) {
            int c = (a + b) * (k - 1);
            a = b;
            b = c;
        }
        return b;

    }

    // 251. 展开二维向量 (Flatten 2D Vector) --plus
    class Vector2D {
        private int[][] vec;
        private int n;
        private int index;
        private int itemIndex;

        public Vector2D(int[][] vec) {
            this.vec = vec;
            this.n = vec.length;

        }

        public int next() {
            int res = vec[index][itemIndex];
            if (++itemIndex == vec[index].length) {
                itemIndex = 0;
                ++index;
            }
            return res;
        }

        public boolean hasNext() {
            while (index < n) {
                if (itemIndex == vec[index].length) {
                    ++index;
                    itemIndex = 0;
                } else {
                    return true;
                }
            }
            return false;
        }
    }

    // 635. 设计日志存储系统 (Design Log Storage System) --plus
    class LogSystem {
        private TreeMap<String, Integer> map;
        private Map<String, Integer> unit;

        public LogSystem() {
            map = new TreeMap<>();
            unit = new HashMap<>();
            // ["Year", "Month", "Day", "Hour", "Minute", "Second"]
            unit.put("Year", 0);
            unit.put("Month", 1);
            unit.put("Day", 2);
            unit.put("Hour", 3);
            unit.put("Minute", 4);
            unit.put("Second", 5);
        }

        public void put(int id, String timestamp) {
            map.put(timestamp, id);
        }

        public List<Integer> retrieve(String start, String end, String granularity) {
            String s1 = formatTime635(start, granularity, false);
            String s2 = formatTime635(end, granularity, true);
            return new ArrayList<>(map.subMap(s1, s2).values());
        }

        private String formatTime635(String time, String granularity, boolean isEnd) {
            // 2017:01:01:23:59:59
            String[] split = time.split(":");
            int[] parseInt = new int[6];
            for (int i = 0; i < 6; ++i) {
                parseInt[i] = Integer.parseInt(split[i]);
                if (i == unit.get(granularity)) {
                    if (isEnd) {
                        ++parseInt[i];
                    }
                    break;
                }
            }
            return String.format("%04d:%02d:%02d:%02d:%02d:%02d", parseInt[0], parseInt[1], parseInt[2], parseInt[3],
                    parseInt[4], parseInt[5]);
        }
    }

    // 270. 最接近的二叉搜索树值 (Closest Binary Search Tree Value) --plus
    private double maxDiff270;
    private int res270;
    private double target270;

    public int closestValue(TreeNode root, double target) {
        maxDiff270 = Math.abs(target - root.val);
        res270 = root.val;
        target270 = target;
        dfs270(root);
        return res270;

    }

    private void dfs270(TreeNode node) {
        if (node == null) {
            return;
        }
        if (Math.abs(node.val - target270) < maxDiff270) {
            maxDiff270 = Math.abs(node.val - target270);
            res270 = node.val;
        }
        dfs270(node.left);
        dfs270(node.right);
    }

    // 408. 有效单词缩写 (Valid Word Abbreviation) --plus
    public boolean validWordAbbreviation(String word, String abbr) {
        int m = word.length();
        int n = abbr.length();
        int i = 0;
        int j = 0;
        while (i < m && j < n) {
            char c = abbr.charAt(j);
            if (Character.isLetter(c)) {
                if (c != word.charAt(i)) {
                    return false;
                }
                ++i;
                ++j;
                continue;
            }
            if (c == '0') {
                return false;
            }
            int count = 0;
            while (j < n && Character.isDigit(abbr.charAt(j))) {
                count = count * 10 + abbr.charAt(j) - '0';
                ++j;
            }
            i += count;
        }
        return i == m && j == n;

    }

    // 170. 两数之和 III - 数据结构设计 (Two Sum III - Data structure design) --plus
    class TwoSum {
        private List<Integer> list;
        private boolean isSorted;

        public TwoSum() {
            list = new ArrayList<>();

        }

        public void add(int number) {
            list.add(number);
            isSorted = false;

        }

        public boolean find(int value) {
            if (!isSorted) {
                isSorted = true;
                Collections.sort(list);
            }
            int left = 0;
            int right = list.size() - 1;
            while (left < right) {
                int sum = list.get(left) + list.get(right);
                if (sum == value) {
                    return true;
                }
                if (sum < value) {
                    ++left;
                } else {
                    --right;
                }
            }
            return false;

        }
    }

    // 170. 两数之和 III - 数据结构设计 (Two Sum III - Data structure design) --plus
    class TwoSum2 {
        private Map<Integer, Integer> map;

        public TwoSum2() {
            map = new HashMap<>();

        }

        public void add(int number) {
            map.put(number, map.getOrDefault(number, 0) + 1);
        }

        public boolean find(int value) {
            for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
                int complement = value - entry.getKey();
                if (complement != entry.getKey()) {
                    if (map.containsKey(complement)) {
                        return true;
                    }
                } else {
                    if (entry.getValue() > 1) {
                        return true;
                    }
                }
            }
            return false;
        }
    }

    // 418. 屏幕可显示句子的数量 (Sentence Screen Fitting) --plus
    public int wordsTyping(String[] sentence, int rows, int cols) {
        int n = sentence.length;
        int[] len = new int[n];
        for (int i = 0; i < n; ++i) {
            len[i] = sentence[i].length();
        }
        int i = 0;
        int j = 0;
        int index = 0;
        int res = 0;
        while (i < rows && j < cols) {
            while (index < n && j + len[index] - 1 < cols) {
                j += len[index++] + 1;
            }
            if (index == n) {
                ++res;
                index = 0;
            }
            if (j + len[index] - 1 >= cols) {
                ++i;
                j = 0;
            }
        }
        return res;
    }

    // 755. 倒水 (Pour Water) --plus
    public int[] pourWater(int[] heights, int volume, int k) {
        int n = heights.length;
        while (volume-- > 0) {
            int i = k - 1;
            int targetIndex = -1;
            while (i >= 0 && heights[i] <= heights[i + 1]) {
                if (heights[i] < heights[i + 1]) {
                    targetIndex = i;
                }
                --i;
            }
            if (targetIndex != -1) {
                ++heights[targetIndex];
                continue;
            }
            i = k + 1;
            targetIndex = -1;
            while (i < n && heights[i - 1] >= heights[i]) {
                if (heights[i - 1] > heights[i]) {
                    targetIndex = i;
                }
                ++i;
            }
            if (targetIndex != -1) {
                ++heights[targetIndex];
                continue;
            }
            ++heights[k];
        }
        return heights;

    }

    // 1060. 有序数组中的缺失元素 (Missing Element in Sorted Array) --plus
    public int missingElement(int[] nums, int k) {
        int n = nums.length;
        int right = n - 1;
        int left = 0;
        int res = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            // int count = nums[mid] - nums[0] - 1 - (mid - 0 - 1);
            int count = nums[mid] - nums[0] - mid;
            if (count < k) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        // return nums[res] + k - (nums[res] - nums[0] - 1 - (res - 0 - 1));
        return k + nums[0] + res;
    }

    // 1061. 按字典序排列最小的等效字符串 (Lexicographically Smallest Equivalent String)
    public String smallestEquivalentString(String s1, String s2, String baseStr) {
        Union1061 union = new Union1061(26);
        int n = s1.length();
        for (int i = 0; i < n; ++i) {
            int index1 = s1.charAt(i) - 'a';
            int index2 = s2.charAt(i) - 'a';
            union.union(index1, index2);
        }
        int[] dic = new int[26];
        for (int i = 0; i < 26; ++i) {
            for (int j = 0; j <= i; ++j) {
                if (union.isConnected(i, j)) {
                    dic[i] = j;
                    break;
                }
            }
        }
        StringBuilder res = new StringBuilder();
        for (char c : baseStr.toCharArray()) {
            int index = c - 'a';
            res.append((char) (dic[index] + 'a'));
        }
        return res.toString();

    }

    public class Union1061 {
        private int[] parent;
        private int[] rank;

        public Union1061(int n) {
            parent = new int[n];
            rank = new int[n];
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

    // 2083. 求以相同字母开头和结尾的子串总数 (Substrings That Begin and End With the Same Letter)
    // --plus
    public long numberOfSubstrings(String s) {
        int[] counts = new int[26];
        long res = 0l;
        for (char c : s.toCharArray()) {
            res += ++counts[c - 'a'];
        }
        return res;

    }

    // 2436. Minimum Split Into Subarrays With GCD Greater Than One --plus
    public int minimumSplits(int[] nums) {
        int res = 0;
        int gcd = 0;
        for (int num : nums) {
            gcd = getGCD2436(gcd, num);
            if (gcd == 1) {
                ++res;
                gcd = num;
            }
        }
        return res + 1;
    }

    private int getGCD2436(int a, int b) {
        return b == 0 ? a : getGCD2436(b, a % b);
    }

    // 6253. 回环句
    public boolean isCircularSentence(String sentence) {
        int n = sentence.length();
        if (sentence.charAt(0) != sentence.charAt(n - 1)) {
            return false;
        }
        for (int i = 0; i < n; ++i) {
            if (sentence.charAt(i) == ' ') {
                if (sentence.charAt(i - 1) != sentence.charAt(i + 1)) {
                    return false;
                }
            }
        }
        return true;

    }

    // 6254. 划分技能点相等的团队
    public long dividePlayers(int[] skill) {
        long res = 0l;
        Arrays.sort(skill);
        int i = 0;
        int j = skill.length - 1;
        int sum = skill[i] + skill[j];
        while (i < j) {
            int curSum = skill[i] + skill[j];
            if (curSum != sum) {
                return -1;
            }
            res += skill[i] * skill[j];
            ++i;
            --j;
        }
        return res;

    }

    // 2492. 两个城市间路径的最小分数 (Minimum Score of a Path Between Two Cities)
    public int minScore(int n, int[][] roads) {
        Map<Integer, List<int[]>> map = new HashMap<>();
        for (int[] road : roads) {
            map.computeIfAbsent(road[0] - 1, k -> new ArrayList<>()).add(new int[] { road[1] - 1, road[2] });
            map.computeIfAbsent(road[1] - 1, k -> new ArrayList<>()).add(new int[] { road[0] - 1, road[2] });

        }
        int res = Integer.MAX_VALUE;
        boolean[] visited = new boolean[n];
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(0);
        visited[0] = true;
        while (!queue.isEmpty()) {
            int node = queue.poll();
            for (int[] neighbor : map.getOrDefault(node, new ArrayList<>())) {
                int nNode = neighbor[0];
                int nDis = neighbor[1];
                if (!visited[nNode]) {
                    visited[nNode] = true;
                    queue.offer(nNode);
                }
                res = Math.min(res, nDis);
            }
        }
        return res;

    }

    // 2492. 两个城市间路径的最小分数 (Minimum Score of a Path Between Two Cities)
    public int minScore2(int n, int[][] roads) {
        Union2492 union = new Union2492(n);
        for (int[] road : roads) {
            union.union(road[0] - 1, road[1] - 1);
        }
        int res = Integer.MAX_VALUE;
        for (int[] road : roads) {
            if (union.isConnected(0, road[0] - 1)) {
                res = Math.min(res, road[2]);
            }
        }
        return res;

    }

    public class Union2492 {
        private int[] rank;
        private int[] parent;

        public Union2492(int n) {
            rank = new int[n];
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                rank[i] = 1;
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

    // 2493. 将节点分成尽可能多的组 (Divide Nodes Into the Maximum Number of Groups)
    private Map<Integer, List<Integer>> graph2493;
    private Union2493 union2493;
    private static final int RED2493 = 1;
    private static final int GREEN2493 = -1;
    private static final int NONE2493 = 0;
    private int n2493;

    public int magnificentSets(int n, int[][] edges) {
        this.n2493 = n;
        graph2493 = new HashMap<>();
        union2493 = new Union2493(n);
        for (int[] edge : edges) {
            graph2493.computeIfAbsent(edge[0] - 1, k -> new ArrayList<>()).add(edge[1] - 1);
            graph2493.computeIfAbsent(edge[1] - 1, k -> new ArrayList<>()).add(edge[0] - 1);
            union2493.union(edge[0] - 1, edge[1] - 1);
        }
        Map<Integer, Integer> counts = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            int groups = getGroups(i);
            if (groups == -1) {
                return -1;
            }
            int root = union2493.getRoot(i);
            counts.merge(root, groups, Integer::max);
        }
        int res = 0;
        for (int c : counts.values()) {
            res += c;
        }
        return res;

    }

    private int getGroups(int node) {
        int[] colors = new int[n2493];
        int res = 0;
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(node);
        colors[node] = RED2493;
        while (!queue.isEmpty()) {
            int size = queue.size();
            ++res;
            for (int i = 0; i < size; ++i) {
                int x = queue.poll();
                for (int y : graph2493.getOrDefault(x, new ArrayList<>())) {
                    if (colors[y] == NONE2493) {
                        colors[y] = colors[x] == RED2493 ? GREEN2493 : RED2493;
                        queue.offer(y);
                    } else if (colors[y] == colors[x]) {
                        return -1;
                    }
                }
            }
        }
        return res;
    }

    class Union2493 {
        private int[] rank;
        private int[] parent;

        public Union2493(int n) {
            rank = new int[n];
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                rank[i] = 1;
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

}
