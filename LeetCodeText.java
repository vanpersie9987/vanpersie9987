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
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;

public class LeetCodeText {
    private int[] nums;
    private int target;
    private int[] nums2;
    private int target2;
    private String word;
    private char[][] board;
    private boolean[][] marked;
    private final int[][] directions = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
    private static String[] transactions = { "alice,20,800,mtv", "alice,50,100,beijing" };

    public static void main(final String[] args) {
        // int[] nums = new int[5];
        // System.out.println(majorityElement(nums));
        // int[] A = new int[] { 12, 24, 8, 32 };
        // int[] B = new int[] { 13, 25, 32, 11 };
        // int[] C = advantageCount(A, B);
        // System.out.println(C);

        // int[] arr = { 3, 2, 4, 1 };
        // pancakeSort(arr);

        // simplifyPath("/a/../../b/../c//.//");
        // reverseWords("a good example ");
        // String s = "2001:0db8:85a3:0:0:8A2E:0370:7334:";
        // validIPAddress(s);
        final List<String> list = new ArrayList<>();
        // // list.add("23:59");
        // list.add("00:00");

        // findMinDifference(list);
        // int[] nums = new int[] { 6, 6, 0, 1, 1, 4, 6 };
        // int a = minDifference2(nums);
        // List<String> list1 = invalidTransactions(transactions);
        // int[] a = { -2, 1, -1, -2, -2 };
        // boolean b = circularArrayLoop(a);
        // int x = reverse(-2147483412);
        // int[] source = { 5, 1, 2, 4, 3 };
        // int[] target = { 1, 5, 4, 2, 3 };
        // int[][] allowedSwaps = { { 0, 4 }, { 4, 2 }, { 1, 3 }, { 1, 4 } };

        // int res = minimumHammingDistance(source, target, allowedSwaps);
        // boolean b = checkZeroOnes("111000");
        // String s = minRemoveToMakeValid2("a)b(c)d");
        // String s = decodeString("3[z]2[2[y]pq4[2[jk]e1[f]]]ef");
        // String[] names = { "John(15)", "Jon(12)", "Chris(13)", "Kris(4)",
        // "Christopher(19)" };
        // String[] synonyms = { "(Jon,John)", "(John,Johnny)", "(Chris,Kris)",
        // "(Chris,Christopher)" };
        // String[] s = trulyMostPopular(names, synonyms);

    }

    // 1.两数之和
    public int[] twoSum(final int[] nums, final int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            int remain = target - nums[i];
            if (map.containsKey(remain)) {
                return new int[] { map.get(remain), i };
            }
            map.put(nums[i], i);
        }
        return null;

    }

    public int maxArea(final int[] height) {
        int max = 0;
        int left = 0;
        int right = height.length - 1;
        while (left < right) {
            max = Math.max(max, Math.min(height[left], height[right]) * (right - left));
            if (height[left] < height[right]) {
                ++left;
            } else {
                --right;
            }
        }
        return max;

    }

    public List<List<Integer>> threeSum(final int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] > 0) {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int target = nums[i] + nums[left] + nums[right];
                if (target == 0) {
                    result.add(Arrays.asList(nums[i], nums[left], nums[right]));
                    while (left < right && nums[left] == nums[left + 1]) {
                        ++left;
                    }
                    while (left < right && nums[right] == nums[right - 1]) {
                        --right;
                    }
                    ++left;
                    --right;
                } else if (target < 0) {
                    ++left;
                } else {
                    --right;
                }
            }
        }
        return result;

    }

    public int threeSumClosest(final int[] nums, final int target) {
        int close = nums[0] + nums[1] + nums[2];
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; ++i) {
            int left = i + 1;
            int right = nums.length - 1;
            while (left < right) {
                int sum = nums[i] + nums[left] + nums[right];
                if (Math.abs(sum - target) < Math.abs(close - target)) {
                    close = sum;
                }

                if (sum == target) {
                    return sum;
                }
                if (sum < target) {
                    ++left;
                } else {
                    --right;
                }
            }
        }
        return close;

    }

    // 18. 四数之和
    public List<List<Integer>> fourSum(final int[] nums, final int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < nums.length - 3; ++i) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            if (nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) {
                break;
            }
            if (nums[i] + nums[nums.length - 1] + nums[nums.length - 2] + nums[nums.length - 3] < target) {
                continue;
            }
            for (int j = i + 1; j < nums.length - 2; ++j) {
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                if (nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) {
                    break;
                }
                if (nums[i] + nums[j] + nums[nums.length - 1] + nums[nums.length - 2] < target) {
                    continue;
                }
                int left = j + 1;
                int right = nums.length - 1;
                while (left < right) {
                    int sum = nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        result.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
                        while (left < right && nums[left] == nums[left + 1]) {
                            ++left;
                        }
                        while (left < right && nums[right] == nums[right - 1]) {
                            --right;
                        }
                        ++left;
                        --right;
                    } else if (sum < target) {
                        ++left;
                    } else {
                        --right;
                    }
                }
            }
        }
        return result;

    }

    // 26. 删除有序数组中的重复项
    public int removeDuplicates(final int[] nums) {
        int index = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] != nums[index]) {
                ++index;
                nums[index] = nums[i];
            }
        }
        return index + 1;

    }

    // 80. 删除有序数组中的重复项 II
    public int removeDuplicatesII(final int[] nums) {
        int index = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (index < 2 || nums[i] != nums[index - 2]) {
                nums[index] = nums[i];
                ++index;
            }
        }
        return index;

    }

    public boolean searchII(final int[] nums, final int target) {
        if (nums == null || nums.length == 0) {
            return false;
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            final int mid = left + ((right - left) >>> 1);
            if (nums[mid] == target) {
                return true;
            }
            if (nums[left] == nums[mid]) {
                ++left;
                continue;
            }
            if (nums[left] < nums[mid]) {
                if (nums[left] <= target && target <= nums[mid]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            } else if (nums[left] > nums[mid]) {
                if (nums[mid] <= target && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        return false;
    }

    // 27. 移除元素
    public int removeElement(final int[] nums, final int val) {
        int index = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] != val) {
                nums[index++] = nums[i];
            }
        }
        return index;

    }

    // 33. 搜索旋转排序数组
    public int search(final int[] nums, final int target) {
        int rotateIndex = findRotateIndex2(nums, 0, nums.length - 1);
        if (nums[rotateIndex] == target) {
            return rotateIndex;
        }
        if (rotateIndex == 0) {
            return binarySearch2(nums, target, 0, nums.length - 1);
        }
        if (nums[0] > target) {
            return binarySearch2(nums, target, rotateIndex, nums.length - 1);
        } else {
            return binarySearch2(nums, target, 0, rotateIndex - 1);
        }

    }

    private int binarySearch2(int[] nums, int target, int left, int right) {
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            }
        }
        return -1;

    }

    private int findRotateIndex2(int[] nums, int left, int right) {
        if (nums[left] < nums[right]) {
            return 0;
        }
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if ((mid + 1 < nums.length) && nums[mid] > nums[mid + 1]) {
                return mid + 1;
            }
            if (nums[left] > nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return 0;

    }

    // 34. 在排序数组中查找元素的第一个和最后一个位置
    public int[] searchRange(final int[] nums, final int target) {
        if (nums == null || nums.length == 0) {
            return new int[] { -1, -1 };
        }
        return new int[] { findLeftEdgeIndex(nums, target), findRightEdgeIndex(nums, target) };
    }

    private int findLeftEdgeIndex(int[] nums, int target) {
        int ans = -1;
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] == target) {
                ans = mid;
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            }
        }
        return ans;

    }

    private int findRightEdgeIndex(int[] nums, int target) {
        int ans = -1;
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] == target) {
                ans = mid;
                left = mid + 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            }
        }
        return ans;

    }

    // 35. 搜索插入位置
    public int searchInsert(final int[] nums, final int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            }
        }
        return left;

    }

    // 42. 接雨水
    public int trap(final int[] height) {
        int res = 0;
        int[] leftMax = new int[height.length];
        int[] rightMax = new int[height.length];
        for (int i = 1; i < height.length; ++i) {
            leftMax[i] = Math.max(leftMax[i - 1], height[i - 1]);
        }
        for (int i = height.length - 2; i >= 0; --i) {
            rightMax[i] = Math.max(rightMax[i + 1], height[i + 1]);
        }
        for (int i = 1; i < height.length - 1; ++i) {
            int min = Math.min(leftMax[i], rightMax[i]);
            if (min > height[i]) {
                res += min - height[i];
            }
        }
        return res;

    }

    // 42. 接雨水
    public int trap2(final int[] height) {
        Stack<Integer> stack = new Stack<>();
        int res = 0;
        for (int i = 0; i < height.length; ++i) {
            while (!stack.isEmpty() && height[stack.peek()] < height[i]) {
                int h = height[stack.pop()];
                if (stack.isEmpty()) {
                    break;
                }
                int distance = i - stack.peek() - 1;
                res += (Math.min(height[stack.peek()], height[i]) - h) * distance;
            }
            stack.push(i);
        }
        return res;

    }

    // 48. 旋转矩阵 / 面试题 01.07. 旋转矩阵
    public void rotate(final int[][] matrix) {
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < i; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        for (int i = 0; i < matrix.length; ++i) {
            reverse48(matrix[i]);
        }

    }

    private void reverse48(int[] matrix) {
        int left = 0;
        int right = matrix.length - 1;
        while (left < right) {
            int temp = matrix[left];
            matrix[left] = matrix[right];
            matrix[right] = temp;
            ++left;
            --right;
        }
    }

    // 53. 最大子序和
    public int maxSubArray(final int[] nums) {
        for (int i = 1; i < nums.length; ++i) {
            nums[i] = Math.max(nums[i], nums[i - 1] + nums[i]);
        }
        return Arrays.stream(nums).max().getAsInt();
    }

    // 面试题 16.17. 连续数列
    public int maxSubArray2(final int[] nums) {
        int pre = nums[0];
        int max = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            pre = Math.max(nums[i], pre + nums[i]);
            max = Math.max(max, pre);
        }
        return max;
    }

    // 54. 螺旋矩阵
    public List<Integer> spiralOrder(final int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        int r1 = 0;
        int r2 = matrix.length - 1;
        int c1 = 0;
        int c2 = matrix[0].length - 1;
        while (r1 <= r2 && c1 <= c2) {
            for (int c = c1; c <= c2; ++c) {
                res.add(matrix[r1][c]);
            }
            for (int r = r1 + 1; r <= r2; ++r) {
                res.add(matrix[r][c2]);
            }
            if (r1 < r2 && c1 < c2) {
                for (int c = c2 - 1; c >= c1; --c) {
                    res.add(matrix[r2][c]);
                }
                for (int r = r2 - 1; r >= r1 + 1; --r) {
                    res.add(matrix[r][c1]);
                }
            }
            ++r1;
            --r2;
            ++c1;
            --c2;
        }
        return res;

    }

    // 剑指 Offer 29. 顺时针打印矩阵
    public int[] spiralOrder2(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return new int[] {};
        }
        int[] res = new int[matrix.length * matrix[0].length];
        int index = 0;
        int r1 = 0;
        int r2 = matrix.length - 1;
        int c1 = 0;
        int c2 = matrix[0].length - 1;
        while (r1 <= r2 && c1 <= c2) {
            for (int c = c1; c <= c2; ++c) {
                res[index++] = matrix[r1][c];
            }
            for (int r = r1 + 1; r <= r2; ++r) {
                res[index++] = matrix[r][c2];
            }
            if (r1 < r2 && c1 < c2) {
                for (int c = c2 - 1; c >= c1; --c) {
                    res[index++] = matrix[r2][c];
                }
                for (int r = r2 - 1; r >= r1 + 1; --r) {
                    res[index++] = matrix[r][c1];
                }
            }
            ++r1;
            --r2;
            ++c1;
            --c2;
        }
        return res;

    }

    // 55. 跳跃游戏
    public boolean canJump(final int[] nums) {
        int lastPos = nums.length - 1;
        for (int i = nums.length - 2; i >= 0; --i) {
            if (nums[i] + i >= lastPos) {
                lastPos = i;
            }
        }
        return lastPos == 0;

    }

    // 45
    public int jump(int[] nums) {
        int step = 0;
        int maxPosition = 0;
        int end = 0;
        for (int i = 0; i < nums.length - 1; ++i) {
            maxPosition = Math.max(maxPosition, i + nums[i]);
            if (i == end) {
                end = maxPosition;
                ++step;
            }

        }
        return step;

    }

    // 1732
    public int largestAltitude(int[] gain) {
        int max = Math.max(gain[0], 0);
        for (int i = 1; i < gain.length; ++i) {
            gain[i] += gain[i - 1];
            max = Math.max(max, gain[i]);
        }
        return max;

    }

    // 1748. 唯一元素的和
    public int sumOfUnique(int[] nums) {
        int[] counts = new int[101];
        for (int num : nums) {
            ++counts[num];
        }
        int sum = 0;
        for (int i = 0; i < counts.length; ++i) {
            if (counts[i] == 1) {
                sum += i;
            }
        }
        return sum;

    }

    // 1748. 唯一元素的和
    public int sumOfUnique2(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int sum = 0;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; ++i) {
            if (i == 0) {
                if (i + 1 < nums.length && nums[i] != nums[i + 1]) {
                    sum += nums[i];
                }
            } else if (i == nums.length - 1) {
                if (i - 1 >= 0 && nums[i] != nums[i - 1]) {
                    sum += nums[i];
                }
            } else {
                if (nums[i - 1] != nums[i] && nums[i] != nums[i + 1]) {
                    sum += nums[i];
                }
            }
        }
        return sum;

    }

    // 56. 合并区间
    public int[][] merge(final int[][] intervals) {
        List<int[]> res = new ArrayList<>();
        Arrays.sort(intervals, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] - o2[0];
            }
        });
        int i = 0;
        while (i < intervals.length) {
            int left = intervals[i][0];
            int right = intervals[i][1];
            while (i + 1 < intervals.length && right >= intervals[i + 1][0]) {
                ++i;
                right = Math.max(right, intervals[i][1]);
            }
            res.add(new int[] { left, right });
            ++i;
        }
        return res.toArray(new int[0][]);

    }

    // 57
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> list = new ArrayList<>();
        int i = 0;
        boolean placed = false;
        while (i < intervals.length) {
            int left = intervals[i][0];
            int right = intervals[i][1];
            if (newInterval[1] >= left && newInterval[0] <= right) {
                placed = true;
                left = Math.min(left, newInterval[0]);
                right = Math.max(right, newInterval[1]);
                while (i + 1 < intervals.length && intervals[i + 1][0] <= right) {
                    ++i;
                    right = Math.max(right, intervals[i][1]);
                }
            }
            ++i;
            list.add(new int[] { left, right });
        }
        if (list.size() == 0 || newInterval[1] < intervals[0][0]) {
            placed = true;
            list.add(0, newInterval);
        } else if (newInterval[0] > intervals[intervals.length - 1][1]) {
            placed = true;
            list.add(newInterval);
        }
        if (!placed) {
            for (int j = 0; j < list.size(); ++j) {
                int[] arr = list.get(j);
                if (arr[1] < newInterval[0] && j + 1 < list.size()) {
                    int[] arr2 = list.get(j + 1);
                    if (arr2[0] > newInterval[1]) {
                        list.add(j + 1, newInterval);
                    }
                }
            }
        }

        return list.toArray(new int[0][]);

    }

    // 59. 螺旋矩阵 II
    public int[][] generateMatrix(final int n) {
        int[][] res = new int[n][n];
        int num = 1;
        int r1 = 0;
        int r2 = n - 1;
        while (r1 <= r2) {
            for (int c = r1; c <= r2; ++c) {
                res[r1][c] = num++;
            }
            for (int r = r1 + 1; r <= r2; ++r) {
                res[r][r2] = num++;
            }
            for (int c = r2 - 1; c >= r1; --c) {
                res[r2][c] = num++;
            }
            for (int r = r2 - 1; r >= r1 + 1; --r) {
                res[r][r1] = num++;
            }
            ++r1;
            --r2;
        }
        return res;

    }

    // 62. 不同路径
    public int uniquePaths(final int m, final int n) {
        // m行数 n列数
        int[] dp = new int[n];
        Arrays.fill(dp, 1);
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[j] += dp[j - 1];
            }
        }
        return dp[n - 1];

    }

    // 63. 不同路径 II
    public int uniquePathsWithObstacles(final int[][] obstacleGrid) {
        if (obstacleGrid[0][0] == 1) {
            return 0;
        }
        obstacleGrid[0][0] = 1;
        for (int i = 1; i < obstacleGrid[0].length; ++i) {
            if (obstacleGrid[0][i - 1] == 1 && obstacleGrid[0][i] == 0) {
                obstacleGrid[0][i] = 1;
            } else {
                obstacleGrid[0][i] = 0;
            }
        }
        for (int i = 1; i < obstacleGrid.length; ++i) {
            if (obstacleGrid[i - 1][0] == 1 && obstacleGrid[i][0] == 0) {
                obstacleGrid[i][0] = 1;
            } else {
                obstacleGrid[i][0] = 0;
            }
        }
        for (int i = 1; i < obstacleGrid.length; ++i) {
            for (int j = 1; j < obstacleGrid[0].length; ++j) {
                if (obstacleGrid[i][j] == 0) {
                    obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
                } else {
                    obstacleGrid[i][j] = 0;
                }
            }
        }
        return obstacleGrid[obstacleGrid.length - 1][obstacleGrid[0].length - 1];

    }

    // 64. 最小路径和
    public int minPathSum(final int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0) {
            return 0;
        }
        for (int i = 1; i < grid.length; ++i) {
            grid[i][0] += grid[i - 1][0];
        }
        for (int j = 1; j < grid[0].length; ++j) {
            grid[0][j] += grid[0][j - 1];
        }
        for (int i = 1; i < grid.length; ++i) {
            for (int j = 1; j < grid[0].length; ++j) {
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[grid.length - 1][grid[0].length - 1];

    }

    // 66. 加一
    public int[] plusOne(final int[] digits) {
        for (int i = digits.length - 1; i >= 0; --i) {
            ++digits[i];
            if (digits[i] % 10 != 0) {
                return digits;
            }
            digits[i] %= 10;
        }
        int[] res = new int[digits.length + 1];
        res[0] = 1;
        return res;

    }

    // 73. 矩阵置零 / 面试题 01.08. 零矩阵
    public void setZeroes(final int[][] matrix) {
        boolean firstRowHasZero = false;
        for (int i = 0; i < matrix[0].length; ++i) {
            if (matrix[0][i] == 0) {
                firstRowHasZero = true;
                break;
            }
        }
        boolean firstColHaszero = false;
        for (int i = 0; i < matrix.length; ++i) {
            if (matrix[i][0] == 0) {
                firstColHaszero = true;
                break;
            }
        }
        for (int i = 1; i < matrix.length; ++i) {
            for (int j = 1; j < matrix[0].length; ++j) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = 0;
                    matrix[i][0] = 0;
                }
            }
        }
        for (int i = 1; i < matrix[0].length; ++i) {
            if (matrix[0][i] == 0) {
                for (int j = 1; j < matrix.length; ++j) {
                    matrix[j][i] = 0;
                }
            }
        }
        for (int i = 1; i < matrix.length; ++i) {
            if (matrix[i][0] == 0) {
                Arrays.fill(matrix[i], 0);
            }
        }
        if (firstRowHasZero) {
            Arrays.fill(matrix[0], 0);
        }
        if (firstColHaszero) {
            for (int i = 0; i < matrix.length; ++i) {
                matrix[i][0] = 0;
            }
        }

    }

    // 74. 搜索二维矩阵
    public boolean searchMatrix(final int[][] matrix, final int target) {
        int left = 0;
        int right = matrix.length * matrix[0].length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            int midNum = matrix[mid / matrix[0].length][mid % matrix[0].length];
            if (midNum == target) {
                return true;
            } else if (midNum < target) {
                left = mid + 1;
            } else if (midNum > target) {
                right = mid - 1;
            }
        }
        return false;

    }

    // 240. 搜索二维矩阵 II
    public boolean searchMatrixII(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int r = 0;
        int c = matrix[0].length - 1;
        while (r < matrix.length && c >= 0) {
            if (target == matrix[r][c]) {
                return true;
            }
            if (target > matrix[r][c]) {
                ++r;
            } else {
                --c;
            }
        }
        return false;

    }

    // 75. 颜色分类
    public void sortColors(final int[] nums) {
        int p2 = nums.length - 1;
        int p = 0;
        int p0 = 0;
        while (p <= p2) {
            if (nums[p] == 2) {
                int temp = nums[p2];
                nums[p2] = nums[p];
                nums[p] = temp;
                --p2;
            } else if (nums[p] == 0) {
                int temp = nums[p];
                nums[p] = nums[p0];
                nums[p0] = temp;
                ++p;
                ++p0;
            } else {
                ++p;
            }
        }

    }

    // 79. 单词搜索
    public boolean exist(final char[][] board, final String word) {
        boolean[][] flag = new boolean[board.length][board[0].length];
        int[][] directions = { { 1, 0 }, { 0, 1 }, { 0, -1 }, { -1, 0 } };
        for (int i = 0; i < board.length; ++i) {
            for (int j = 0; j < board[0].length; ++j) {
                if (containsWord(flag, directions, board, word, i, j, 0)) {
                    return true;
                }
            }
        }
        return false;

    }

    private boolean containsWord(boolean[][] flag, int[][] directions, char[][] board, String word, int i, int j,
            int start) {
        if (start == word.length() - 1) {
            return board[i][j] == word.charAt(start);
        }
        if (word.charAt(start) == board[i][j]) {
            flag[i][j] = true;
            for (int k = 0; k < directions.length; ++k) {
                int newI = i + directions[k][0];
                int newJ = j + directions[k][1];
                if (judgeEdge(newI, newJ, board) && !flag[newI][newJ]
                        && containsWord(flag, directions, board, word, newI, newJ, start + 1)) {
                    return true;
                }
            }
            flag[i][j] = false;
        }
        return false;
    }

    private boolean judgeEdge(int newI, int newJ, char[][] board) {
        return newI >= 0 && newI < board.length && newJ >= 0 && newJ < board[0].length;

    }

    // 1295. 统计位数为偶数的数字
    public int findNumbers(final int[] nums) {
        int count = 0;
        for (int num : nums) {
            if (((int) (Math.log10(num) + 1)) % 2 == 0) {
                ++count;
            }
        }
        return count;

    }

    public int largestRectangleArea(final int[] heights) {
        int max = 0;
        if (heights == null || heights.length == 0) {
            return max;
        }
        for (int i = 0; i < heights.length; ++i) {
            for (int j = i; j < heights.length; ++j) {
                int minHeight = Integer.MAX_VALUE;
                for (int k = i; k < j; ++k) {
                    minHeight = Math.min(minHeight, heights[k]);
                }
                max = Math.max(max, minHeight * (j - i + 1));
            }
        }
        return max;
    }

    public int largestRectangleArea2(final int[] heights) {
        int max = 0;
        if (heights == null) {
            return max;
        }
        for (int i = 0; i < heights.length; ++i) {
            int minHeight = Integer.MAX_VALUE;
            for (int j = i; j < heights.length; ++j) {
                minHeight = Math.min(minHeight, heights[j]);
                max = Math.max(max, minHeight * (j - i + 1));
            }
        }
        return max;
    }

    // 84. 柱状图中最大的矩形
    public int largestRectangleArea3(final int[] heights) {
        int max = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for (int i = 0; i < heights.length; ++i) {
            while (stack.peek() != -1 && heights[i] <= heights[stack.peek()]) {
                int h = heights[stack.pop()];
                max = Math.max(max, (i - stack.peek() - 1) * h);
            }
            stack.push(i);
        }
        while (stack.peek() != -1) {
            max = Math.max(max, heights[stack.pop()] * (heights.length - stack.peek() - 1));
        }
        return max;
    }

    public int largestRectangleArea4(final int[] heights) {
        if (heights == null) {
            return 0;
        }
        return calculateArea(heights, 0, heights.length - 1);

    }

    private int calculateArea(final int[] heights, final int start, final int end) {
        if (end < start) {
            return 0;
        }
        int minI = start;
        for (int i = start; i <= end; ++i) {
            if (heights[i] < heights[minI]) {
                minI = i;
            }
        }
        return Math.max(heights[minI] * (end - start + 1),
                Math.max(calculateArea(heights, start, minI - 1), calculateArea(heights, minI + 1, end)));

    }

    // 88. 合并两个有序数组 / 面试题 10.01. 合并排序的数组
    public void merge(final int[] nums1, final int m, final int[] nums2, final int n) {
        int p1 = m - 1;
        int p2 = n - 1;
        int p = m + n - 1;
        while (p1 >= 0 && p2 >= 0) {
            nums1[p--] = nums1[p1] > nums2[p2] ? nums1[p1--] : nums2[p2--];
        }
        System.arraycopy(nums2, 0, nums1, 0, p2 + 1);

    }

    // 118 杨辉三角
    public List<List<Integer>> generate(final int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        if (numRows == 0) {
            return res;
        }
        List<Integer> row1 = new ArrayList<>();
        row1.add(1);
        res.add(row1);
        for (int i = 1; i < numRows; ++i) {
            List<Integer> row = new ArrayList<>();
            row.add(1);
            List<Integer> preRow = res.get(i - 1);
            for (int j = 1; j < preRow.size(); ++j) {
                row.add(preRow.get(j - 1) + preRow.get(j));
            }
            row.add(1);
            res.add(row);
        }
        return res;

    }

    // 119. 杨辉三角 II
    public List<Integer> getRow(final int rowIndex) {
        List<Integer> res = new ArrayList<>();
        res.add(1);
        for (int i = 0; i < rowIndex; ++i) {
            for (int j = i; j > 0; --j) {
                res.set(j, res.get(j) + res.get(j - 1));
            }
            res.add(1);
        }
        return res;

    }

    // 120. 三角形最小路径和
    public int minimumTotal(final List<List<Integer>> triangle) {
        int[] dp = new int[triangle.size() + 1];
        for (int i = triangle.size() - 1; i >= 0; --i) {
            for (int j = 0; j < triangle.get(i).size(); ++j) {
                dp[j] = Math.min(dp[j + 1], dp[j]) + triangle.get(i).get(j);
            }
        }
        return dp[0];

    }

    // 121. 买卖股票的最佳时机
    public int maxProfit(final int[] prices) {
        int maxProfit = 0;
        int low = prices[0];
        for (int i = 1; i < prices.length; ++i) {
            if (prices[i] > low) {
                maxProfit = Math.max(maxProfit, prices[i] - low);
            } else {
                low = prices[i];
            }
        }
        return maxProfit;

    }

    // 122. 买卖股票的最佳时机 II
    public int maxProfit2(final int[] prices) {
        int max = 0;
        for (int i = 1; i < prices.length; ++i) {
            if (prices[i] > prices[i - 1]) {
                max += prices[i] - prices[i - 1];
            }
        }
        return max;

    }

    // 123. 买卖股票的最佳时机 III
    public int maxProfit3(int[] prices) {
        int buy1 = -prices[0];
        int sell1 = 0;
        int buy2 = -prices[0];
        int sell2 = 0;
        for (int i = 0; i < prices.length; ++i) {
            buy1 = Math.max(buy1, -prices[i]);
            sell1 = Math.max(sell1, buy1 + prices[i]);
            buy2 = Math.max(buy2, sell1 - prices[i]);
            sell2 = Math.max(sell2, buy2 + prices[i]);
        }
        return sell2;

    }

    public int maxProfit4(final int[] prices, final int fee) {
        if (prices == null || prices.length == 0) {
            return 0;
        }
        int min = prices[0];
        int profit = 0;
        for (int i = 0; i < prices.length; ++i) {
            if (min > prices[i]) {
                min = prices[i];
            } else if (prices[i] > min + fee) {
                profit += prices[i] - min - fee;
                min = prices[i] - fee;
            }
        }
        return profit;

    }

    // 128. 最长连续序列
    public int longestConsecutive(final int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        Set<Integer> set = new HashSet<>();
        int max = 0;
        for (int num : nums) {
            set.add(num);
        }
        for (int i = 0; i < nums.length; ++i) {
            if (!set.contains(nums[i] - 1)) {
                int curNum = nums[i];
                int count = 1;
                while (set.contains(curNum + 1)) {
                    ++curNum;
                    ++count;
                }
                max = Math.max(max, count);
            }
        }
        return max;

    }

    // 128. 最长连续序列
    public int longestConsecutive2(final int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        Map<Integer, Integer> map = new HashMap<>();
        int index = 0;
        for (int num : nums) {
            if (!map.containsKey(num)) {
                map.put(num, index++);
            }
        }
        Union128 union = new Union128(index);
        for (int key : map.keySet()) {
            if (map.containsKey(key + 1)) {
                union.union(map.get(key), map.get(key + 1));
            }
            if (map.containsKey(key - 1)) {
                union.union(map.get(key), map.get(key - 1));
            }
        }
        return union.getMax();

    }

    public class Union128 {
        private int[] parent;
        private int[] size;
        private int[] rank;
        private int max;

        public Union128(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            size = new int[n];
            Arrays.fill(size, 1);
            rank = new int[n];
            Arrays.fill(rank, 1);
            max = 1;
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
                size[root2] += size[root1];
                max = Math.max(max, size[root2]);
            } else if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
                size[root1] += size[root2];
                max = Math.max(max, size[root1]);
            } else {
                parent[root1] = root2;
                size[root2] += size[root1];
                ++rank[root2];
                max = Math.max(max, size[root2]);
            }
        }

        public int getMax() {
            return max;
        }
    }

    // 152. 乘积最大子数组
    public int maxProduct(final int[] nums) {
        int max = Integer.MIN_VALUE;
        int iMax = 1;
        int iMin = 1;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] < 0) {
                int temp = iMax;
                iMax = iMin;
                iMin = temp;
            }
            iMax = Math.max(nums[i], iMax * nums[i]);
            iMin = Math.min(nums[i], iMin * nums[i]);
            max = Math.max(max, iMax);
        }
        return max;

    }

    // 153. 寻找旋转排序数组中的最小值
    public int findMin(final int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (mid + 1 < nums.length && nums[mid] > nums[mid + 1]) {
                return nums[mid + 1];
            }
            if (nums[left] > nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return nums[0];

    }

    // 167.两数之和 II-输入有序数组
    public int[] twoSum2(final int[] numbers, final int target) {
        int left = 0;
        int right = numbers.length - 1;
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return new int[] { left + 1, right + 1 };
            } else if (sum < target) {
                ++left;
            } else if (sum > target) {
                --right;
            }
        }
        return new int[] { -1, -1 };

    }

    public int majorityElemen3(final int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];

    }

    // 189. 旋转数组
    public void rotate(final int[] nums, final int k) {
        int finalK = k % nums.length;
        reverse189(nums, 0, nums.length - 1);
        reverse189(nums, 0, finalK - 1);
        reverse189(nums, finalK, nums.length - 1);
    }

    private void reverse189(int[] nums, int left, int right) {
        while (left < right) {
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            ++left;
            --right;
        }
    }

    // 189. 旋转数组
    public void rotate2(final int[] nums, final int k) {
        int[] arrCopy = nums.clone();
        for (int i = 0; i < nums.length; ++i) {
            arrCopy[(i + k) % nums.length] = nums[i];
        }
        System.arraycopy(arrCopy, 0, nums, 0, nums.length);

    }

    public int minSubArrayLen(final int s, final int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        int min = Integer.MAX_VALUE;
        int sum = 0;
        int left = 0;
        int right = 0;
        while (right < nums.length) {
            sum += nums[right];
            while (sum >= s) {
                min = Math.min(min, right - left + 1);
                sum -= nums[left];
                ++left;
            }
            ++right;
        }
        return min == Integer.MAX_VALUE ? 0 : min;

    }

    // 217. 存在重复元素
    public boolean containsDuplicate(final int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; ++i) {
            if (set.contains(nums[i])) {
                return true;
            }
            set.add(nums[i]);
        }
        return false;

    }

    // 217. 存在重复元素
    public boolean containsDuplicate2(final int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }
        return set.size() != nums.length;

    }

    // 219. 存在重复元素 II
    public boolean containsNearbyDuplicate(final int[] nums, final int k) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; ++i) {
            if (set.contains(nums[i])) {
                return true;
            }
            set.add(nums[i]);
            if (set.size() > k) {
                set.remove(nums[i - k]);
            }
        }
        return false;

    }

    // 228. 汇总区间
    public static List<String> summaryRanges(final int[] nums) {
        List<String> res = new ArrayList<>();
        int left = 0;
        int right = 1;
        while (right < nums.length) {
            while (right < nums.length && nums[right] - nums[right - 1] == 1) {
                ++right;
            }
            if (right - 1 == left) {
                res.add(String.valueOf(nums[left]));
            } else {
                res.add(nums[left] + "->" + nums[right - 1]);
            }
            left = right;
            ++right;
        }
        if (left == nums.length - 1) {
            res.add(String.valueOf(nums[left]));
        }
        return res;

    }

    // 229. 求众数 II
    public static List<Integer> majorityElement2(final int[] nums) {
        List<Integer> res = new ArrayList<>();
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getValue() > (nums.length / 3)) {
                res.add(entry.getKey());
            }
        }
        return res;

    }

    // 229. 求众数 II
    public static List<Integer> majorityElement3(final int[] nums) {
        int majorityA = nums[0];
        int majorityB = nums[0];
        int countA = 0;
        int countB = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == majorityA) {
                ++countA;
            } else if (nums[i] == majorityB) {
                ++countB;
            } else {
                if (countA == 0) {
                    majorityA = nums[i];
                    ++countA;
                } else if (countB == 0) {
                    majorityB = nums[i];
                    ++countB;
                } else {
                    --countA;
                    --countB;
                }
            }
        }
        countA = 0;
        countB = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == majorityA) {
                ++countA;
            } else if (nums[i] == majorityB) {
                ++countB;
            }
        }

        List<Integer> res = new ArrayList<>();
        if (countA > nums.length / 3) {
            res.add(majorityA);
        }
        if (countB > nums.length / 3) {
            res.add(majorityB);
        }
        return res;

    }

    // 238. 除自身以外数组的乘积
    public static int[] productExceptSelf(final int[] nums) {
        int k = 1;
        int[] result = new int[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            result[i] = k;
            k *= nums[i];
        }
        k = 1;
        for (int i = nums.length - 1; i >= 0; --i) {
            result[i] *= k;
            k *= nums[i];
        }
        return result;

    }

    // 268. 丢失的数字
    public static int missingNumber(final int[] nums) {
        int missing = nums.length;
        for (int i = 0; i < nums.length; ++i) {
            missing ^= i * nums[i];
        }
        return missing;

    }

    // 268. 丢失的数字
    public static int missingNumber2(final int[] nums) {
        final int n = nums.length;
        final int sumExpect = (0 + n) * (n + 1) / 2;
        int sumReal = Arrays.stream(nums).sum();
        return sumExpect - sumReal;

    }

    // 283. 移动零
    public static void moveZeroes(final int[] nums) {
        int left = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] != 0) {
                int temp = nums[i];
                nums[i] = nums[left];
                nums[left] = temp;
                ++left;
            }
        }

    }

    // 287. 寻找重复数
    public static int findDuplicate(final int[] nums) {
        int ans = 0;
        int left = 1;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            int count = 0;
            for (int num : nums) {
                if (num <= mid) {
                    ++count;
                }
            }
            if (count <= mid) {
                left = mid + 1;
            } else {
                ans = mid;
                right = mid - 1;
            }
        }
        return ans;

    }

    public boolean hasCycle(ListNode head) {
        Set<ListNode> set = new HashSet<>();
        if (head == null || head.next == null) {
            return false;
        }
        while (head != null) {
            if (!set.contains(head)) {
                set.add(head);
                head = head.next;
            } else {
                return true;
            }
        }
        return false;

    }

    public boolean hasCycle2(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
            if (fast == null || fast.next == null) {
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;

    }

    public boolean hasCycle3(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head;
        do {
            if (fast == null || fast.next == null) {
                return false;
            }
            fast = fast.next.next;
            slow = slow.next;
        } while (slow != fast);
        return true;

    }

    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) {
            return null;
        }
        ListNode fast = head.next;
        ListNode slow = head;
        while (fast != slow) {
            if (fast == null || fast.next == null) {
                return null;
            }
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;

    }

    class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }
    }

    public void gameOfLife(final int[][] board) {
        // 1--->1 1
        // 0--->0 0
        // 1--->0 2
        // 0--->1 -1
        for (int i = 0; i < board.length; ++i) {
            for (int j = 0; j < board[0].length; ++j) {
                locNumber(board, i, j);
            }
        }
        for (int i = 0; i < board.length; ++i) {
            for (int j = 0; j < board[0].length; ++j) {
                if (board[i][j] == 1 || board[i][j] == -1) {
                    board[i][j] = 1;
                } else {
                    board[i][j] = 0;
                }
            }
        }
    }

    private void locNumber(int[][] board, int i, int j) {
        int count = 0;
        int left = Math.max(j - 1, 0);
        int right = Math.min(j + 1, board[0].length - 1);
        int top = Math.max(i - 1, 0);
        int bottom = Math.min(i + 1, board.length - 1);
        for (int x = top; x <= bottom; ++x) {
            for (int y = left; y <= right; ++y) {
                if (board[x][y] == 1 || board[x][y] == 2) {
                    ++count;
                }
            }
        }
        if (board[i][j] == 1) {
            if (count < 3 || count > 4) {
                board[i][j] = 2;
            } else {
                board[i][j] = 1;

            }
        } else {
            if (count == 3) {
                board[i][j] = -1;
            }

        }
    }

    // 414. 第三大的数
    public int thirdMax3(final int[] nums) {
        // // 排除
        // //
        // Integer.MIN_VALUE，Integer.MIN_VALUE，Integer.MIN_VALUE，Integer.MIN_VALUE，Integer.MIN_VALUE。。。
        // int max = Integer.MIN_VALUE;
        // for (int i = 0; i < nums.length; ++i) {
        // if (nums[i] > max) {
        // max = nums[i];
        // }
        // }
        // if (max == Integer.MIN_VALUE) {
        // return max;
        // }

        // // 排除
        // //
        // 5，5，5，5，5，Integer.MIN_VALUE，Integer.MIN_VALUE，Integer.MIN_VALUE，Integer.MIN_VALUE，Integer.MIN_VALUE。。。

        // boolean flag = false;
        // int max2 = Integer.MIN_VALUE;
        // for (int i = 0; i < nums.length; ++i) {
        // if (nums[i] == Integer.MIN_VALUE || nums[i] == max) {
        // continue;
        // }
        // if (nums[i] > max2) {
        // max2 = nums[i];
        // }
        // flag = true;
        // }
        // if (!flag) {
        // return max;
        // }

        // // 如果走完下面的循环 flag还是false 说明形如：3，3，3，3，3，3，5，5，5，5，5
        // // 如果为true 说明最少有三个数
        // flag = false;
        // int max3 = Integer.MIN_VALUE;
        // for (int i = 0; i < nums.length; ++i) {
        // if (nums[i] == max || nums[i] == max2) {
        // continue;
        // }
        // if (nums[i] > max3) {
        // max3 = nums[i];
        // }
        // flag = true;
        // }
        // if (!flag) {
        // return max;
        // } else {
        // return max3;
        // }

        int max = Integer.MIN_VALUE;
        boolean flag = false;
        for (int num : nums) {
            if (num > max) {
                flag = true;
                max = num;
            }
        }
        if (!flag) {
            return max;
        }
        flag = false;
        int max2 = Integer.MIN_VALUE;
        for (int num : nums) {
            if (num == max) {
                continue;
            }
            if (num > max2) {
                flag = true;
                max2 = num;
            }
        }
        if (!flag) {
            return max;
        }
        int max3 = Integer.MIN_VALUE;
        flag = false;
        for (int num : nums) {
            if (num == max || num == max2) {
                continue;
            }
            flag = true;
            if (num > max3) {
                max3 = num;
            }
        }
        if (!flag) {
            return max;
        } else {
            return max3;
        }

    }

    // 485. 最大连续 1 的个数
    public int findMaxConsecutiveOnes(final int[] nums) {
        int max = 0;
        int count = 0;
        for (int num : nums) {
            if (num == 1) {
                ++count;
            } else {
                max = Math.max(max, count);
                count = 0;
            }
        }
        return Math.max(max, count);

    }

    public int findPoisonedDuration(final int[] timeSeries, final int duration) {
        int sum = 0;
        if (timeSeries == null || timeSeries.length == 0) {
            return sum;
        }
        for (int i = 0; i < timeSeries.length - 1; ++i) {
            if (timeSeries[i + 1] - timeSeries[i] >= duration) {
                sum += duration;
            } else {
                sum += timeSeries[i + 1] - timeSeries[i];
            }
        }
        sum += duration;
        return sum;

    }

    // 532. 数组中的 k-diff 数对
    public int findPairs(final int[] nums, final int k) {
        if (nums == null || nums.length <= 1) {
            return 0;

        }
        int count = 0;
        Arrays.sort(nums);
        int left = 0;
        int right = 1;
        while (right < nums.length) {
            if (nums[right] - nums[left] > k) {
                ++left;
            } else if (left == right || nums[right] - nums[left] < k) {
                ++right;
            } else {
                ++count;
                while (right + 1 < nums.length && nums[right] == nums[right + 1]) {
                    ++right;
                }
                ++right;
            }

        }
        return count;

    }

    // 561. 数组拆分 I
    public int arrayPairSum(final int[] nums) {
        int sum = 0;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; i += 2) {
            sum += nums[i];
        }
        return sum;

    }

    // 565. 数组嵌套
    public int arrayNesting(final int[] nums) {
        int max = 1;
        int count = 0;
        boolean[] visited = new boolean[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            int index = i;
            while (!visited[index]) {
                visited[index] = true;
                index = nums[index];
                ++count;
            }
            max = Math.max(max, count);
            count = 0;

        }
        return max;

    }

    // 560. 和为K的子数组
    public int subarraySum(final int[] nums, final int k) {
        int count = 0;
        for (int i = 0; i < nums.length; ++i) {
            int sum = 0;
            for (int j = i; j >= 0; --j) {
                sum += nums[j];
                if (sum == k) {
                    ++count;
                }
            }
        }
        return count;

    }

    // 495. 提莫攻击
    public int findPoisonedDuration2(final int[] timeSeries, final int duration) {
        if (timeSeries == null || timeSeries.length == 0) {
            return 0;
        }
        int time = 0;
        for (int i = 1; i < timeSeries.length; ++i) {
            time += Math.min(timeSeries[i] - timeSeries[i - 1], duration);
        }
        time += duration;
        return time;

    }

    // 448. 找到所有数组中消失的数字
    public List<Integer> findDisappearedNumbers(final int[] nums) {
        for (int i = 0; i < nums.length; ++i) {
            if (nums[Math.abs(nums[i]) - 1] > 0) {
                nums[Math.abs(nums[i]) - 1] *= -1;
            }

        }
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] > 0) {
                list.add(i + 1);
            }
        }
        return list;

    }

    // 442. 数组中重复的数据
    public List<Integer> findDuplicates(int[] nums) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < nums.length; ++i) {
            if (nums[Math.abs(nums[i]) - 1] < 0) {
                list.add(Math.abs(nums[i]));
            } else {
                nums[Math.abs(nums[i]) - 1] *= -1;
            }
        }
        return list;

    }

    // 287. 寻找重复数
    public int findDuplicate2(final int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < nums.length; ++i) {
            if (set.contains(nums[i])) {
                return nums[i];
            }
            set.add(nums[i]);
        }
        return nums[0];
    }

    // 566. 重塑矩阵
    public int[][] matrixReshape(final int[][] nums, final int r, final int c) {
        if (r == 0 || c == 0 || nums == null || nums.length == 0 || nums[0].length == 0) {
            return null;
        }
        if (r * c != nums.length * nums[0].length) {
            return nums;
        }
        int index = 0;
        int[][] ans = new int[r][c];
        for (int i = 0; i < nums.length; ++i) {
            for (int j = 0; j < nums[0].length; ++j) {
                ans[index / c][index % c] = nums[i][j];
                ++index;
            }
        }
        return ans;

    }

    // 581. 最短无序连续子数组
    public int findUnsortedSubarray(final int[] nums) {
        int[] arrCopy = nums.clone();
        int left = nums.length - 1;
        int right = 0;
        Arrays.sort(arrCopy);
        for (int i = 0; i < nums.length; ++i) {
            if (arrCopy[i] != nums[i]) {
                left = Math.min(left, i);
                right = Math.max(right, i);
            }
        }
        return right > left ? right - left + 1 : 0;

    }

    // 581. 最短无序连续子数组
    public int findUnsortedSubarray2(final int[] nums) {
        int left = nums.length - 1;
        int right = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < nums.length; ++i) {
            while (!stack.isEmpty() && nums[stack.peek()] > nums[i]) {
                left = Math.min(left, stack.pop());
            }
            stack.push(i);
        }
        stack.clear();
        for (int i = nums.length - 1; i >= 0; --i) {
            while (!stack.isEmpty() && nums[stack.peek()] < nums[i]) {
                right = Math.max(right, stack.pop());
            }
            stack.push(i);
        }
        return right > left ? right - left + 1 : 0;

    }

    // 581. 最短无序连续子数组
    public int findUnsortedSubarray3(final int[] nums) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] < nums[i - 1]) {
                min = Math.min(min, nums[i]);
            }
        }
        for (int i = nums.length - 2; i >= 0; --i) {
            if (nums[i] > nums[i + 1]) {
                max = Math.max(max, nums[i]);
            }
        }
        int left = nums.length - 1;
        int right = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] > min) {
                left = i;
                break;
            }
        }
        for (int i = nums.length - 1; i >= 0; --i) {
            if (nums[i] < max) {
                right = i;
                break;
            }
        }
        return right > left ? right - left + 1 : 0;

    }

    public int countConsistentStrings(String allowed, String[] words) {
        int count = 0;
        for (String word : words) {
            boolean flag = false;
            for (char a : word.toCharArray()) {
                if (!allowed.contains(String.valueOf(a))) {
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                ++count;
            }
        }
        return count;

    }

    // 605. 种花问题
    public boolean canPlaceFlowers(final int[] flowerbed, int n) {
        if (flowerbed.length == 1) {
            if (flowerbed[0] == 0) {
                --n;
            }
            return n <= 0;
        }
        if (flowerbed[0] == 0 && flowerbed[1] == 0) {
            flowerbed[0] = 1;
            --n;
        }
        for (int i = 1; i < flowerbed.length - 1; ++i) {
            if (flowerbed[i - 1] == 0 && flowerbed[i] == 0 && flowerbed[i + 1] == 0) {
                flowerbed[i] = 1;
                --n;
            }
        }
        if (flowerbed[flowerbed.length - 1] == 0 && flowerbed[flowerbed.length - 2] == 0) {
            flowerbed[flowerbed.length - 1] = 1;
            --n;
        }
        return n <= 0;

    }

    // 611. 有效三角形的个数
    public int triangleNumber(final int[] nums) {
        int count = 0;
        for (int i = 0; i < nums.length - 2; ++i) {
            for (int j = i + 1; j < nums.length - 1; ++j) {
                for (int k = j + 1; k < nums.length; ++k) {
                    if (nums[i] + nums[j] > nums[k] && nums[j] + nums[k] > nums[i] && nums[k] + nums[i] > nums[j]) {
                        ++count;
                    }
                }
            }
        }
        return count;

    }

    // 611. 有效三角形的个数
    public int triangleNumber2(final int[] nums) {
        int count = 0;
        Arrays.sort(nums);
        for (int i = nums.length - 1; i >= 0; --i) {
            int left = 0;
            int right = i - 1;
            while (left < right) {
                if (nums[left] + nums[right] > nums[i]) {
                    count += right - left;
                    --right;
                } else {
                    ++left;
                }
            }
        }
        return count;

    }

    // 621. 任务调度器
    public int leastInterval(final char[] tasks, final int n) {
        int[] counts = new int[26];
        for (char task : tasks) {
            ++counts[task - 'A'];
        }
        Arrays.sort(counts);
        int max = counts[25] - 1;
        int slots = n * max;
        for (int i = counts.length - 2; i >= 0; --i) {
            slots -= Math.min(counts[i], max);
        }
        return slots >= 0 ? slots + tasks.length : tasks.length;

    }

    // 628. 三个数的最大乘积
    public int maximumProduct(final int[] nums) {
        Arrays.sort(nums);
        return Math.max(nums[0] * nums[1] * nums[nums.length - 1],
                nums[nums.length - 1] * nums[nums.length - 3] * nums[nums.length - 2]);

    }

    // 628. 三个数的最大乘积
    public int maximumProduct2(final int[] nums) {
        int min1 = Integer.MAX_VALUE;
        int min2 = Integer.MAX_VALUE;
        int max1 = Integer.MIN_VALUE;
        int max2 = Integer.MIN_VALUE;
        int max3 = Integer.MIN_VALUE;
        for (int num : nums) {
            if (num >= max1) {
                max3 = max2;
                max2 = max1;
                max1 = num;
            } else if (num >= max2) {
                max3 = max2;
                max2 = num;
            } else if (num >= max3) {
                max3 = num;
            }

            if (num <= min1) {
                min2 = min1;
                min1 = num;
            } else if (num <= min2) {
                min2 = num;
            }
        }
        return Math.max(max1 * max2 * max3, max1 * min1 * min2);

    }

    // 643. 子数组最大平均数 I
    public double findMaxAverage(final int[] nums, final int k) {
        int curMax = 0;
        for (int i = 0; i < k; ++i) {
            curMax += nums[i];
        }
        int max = curMax;
        for (int i = k; i < nums.length; ++i) {
            curMax += nums[i] - nums[i - k];
            max = Math.max(max, curMax);
        }
        return (double) max / k;

    }

    // 661. 图片平滑器
    public int[][] imageSmoother(final int[][] M) {
        int[][] result = new int[M.length][M[0].length];
        for (int i = 0; i < M.length; ++i) {
            for (int j = 0; j < M[0].length; ++j) {
                int left = Math.max(j - 1, 0);
                int right = Math.min(j + 1, M[0].length - 1);
                int top = Math.max(i - 1, 0);
                int bottom = Math.min(i + 1, M.length - 1);
                int count = 0;
                for (int x = top; x <= bottom; ++x) {
                    for (int y = left; y <= right; ++y) {
                        result[i][j] += M[x][y];
                        ++count;
                    }
                }
                result[i][j] /= count;
            }
        }
        return result;

    }

    // 665. 非递减数列
    public boolean checkPossibility(final int[] nums) {
        if (nums.length <= 2) {
            return true;
        }
        int n = 1;
        if (nums[0] > nums[1]) {
            nums[0] = nums[1];
            --n;
        }
        for (int i = 1; i < nums.length - 1; ++i) {
            if (nums[i] > nums[i + 1]) {
                --n;
                if (n < 0) {
                    return false;
                }
                if (nums[i + 1] < nums[i - 1]) {
                    nums[i + 1] = nums[i];
                } else {
                    nums[i] = nums[i + 1];
                }
            }
        }
        return true;

    }

    // n = 8
    // k == 1 // 1 2 3 4 5 6 7 8
    // k == 2 // 1 8 7 6 5 4 3 2
    // k == 3 // 1 8 2 3 4 5 6 7

    // 667.优美的排列 II
    public int[] constructArray(final int n, final int k) {
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            res[i] = i + 1;
        }
        for (int i = 1; i < k; ++i) {
            reverse667(res, i, n - 1);
        }
        return res;
    }

    private void reverse667(int[] nums, int left, int right) {
        while (left < right) {
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            ++left;
            --right;
        }
    }

    // 674. 最长连续递增序列
    public int findLengthOfLCIS(final int[] nums) {
        int max = 1;
        int count = 1;
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] > nums[i - 1]) {
                ++count;
            } else {
                max = Math.max(max, count);
                count = 1;
            }
        }
        return Math.max(max, count);

    }

    // 695. 岛屿的最大面积
    public int maxAreaOfIsland(final int[][] grid) {
        int max = 0;
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                max = Math.max(max, getMaxArea695(grid, i, j));
            }
        }
        return max;

    }

    private int getMaxArea695(int[][] grid, int i, int j) {
        int count = 0;
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length) {
            return 0;
        }
        if (grid[i][j] == 0) {
            return 0;
        }
        grid[i][j] = 0;
        ++count;

        count += getMaxArea695(grid, i - 1, j);
        count += getMaxArea695(grid, i + 1, j);
        count += getMaxArea695(grid, i, j - 1);
        count += getMaxArea695(grid, i, j + 1);
        return count;
    }

    // 695. 岛屿的最大面积(并查集)
    public int maxAreaOfIsland2(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Union695 union = new Union695(m * n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    if (i > 0 && grid[i - 1][j] == 1) {
                        union.union(getIndex695(n, i, j), getIndex695(n, i - 1, j));
                    }
                    if (j > 0 && grid[i][j - 1] == 1) {
                        union.union(getIndex695(n, i, j), getIndex695(n, i, j - 1));
                    }
                }
            }
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    int index = getIndex695(n, i, j);
                    int root = union.getRoot(index);
                    map.put(root, map.getOrDefault(root, 0) + 1);
                }
            }
        }
        if (map.isEmpty()) {
            return 0;
        }
        return Collections.max(map.values());

    }

    private int getIndex695(int n, int i, int j) {
        return i * n + j;
    }

    public class Union695 {
        private int[] rank;
        private int[] parent;

        public Union695(int n) {
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

    // 697. 数组的度
    public int findShortestSubArray2(final int[] nums) {
        // 计算数组中每一个值 第一次出现和最后一次出现的位置
        Map<Integer, Integer> left = new HashMap<>();
        Map<Integer, Integer> right = new HashMap<>();
        Map<Integer, Integer> count = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            if (!left.containsKey(nums[i])) {
                left.put(nums[i], i);
            }
            right.put(nums[i], i);
            count.put(nums[i], count.getOrDefault(nums[i], 0) + 1);
        }
        int degree = Collections.max(count.values());
        int res = nums.length;
        for (int num : count.keySet()) {
            if (count.get(num) == degree) {
                res = Math.min(res, right.get(num) - left.get(num) + 1);
            }
        }
        return res;

    }

    // 713. 乘积小于K的子数组
    public int numSubarrayProductLessThanK(final int[] nums, final int k) {
        if (k <= 1) {
            return 0;
        }
        int res = 0;
        int left = 0;
        int val = 1;
        for (int i = 0; i < nums.length; ++i) {
            val *= nums[i];
            while (val >= k) {
                val /= nums[left];
                ++left;
            }
            res += i - left + 1;
        }
        return res;

    }

    // 717. 1比特与2比特字符
    public boolean isOneBitCharacter(final int[] bits) {
        int i = 0;
        while (i < bits.length - 1) {
            i += bits[i] + 1;
        }
        return i == bits.length - 1;

    }

    // 717. 1比特与2比特字符
    public boolean isOneBitCharacter2(final int[] bits) {
        int count = 0;
        for (int i = bits.length - 2; i >= 0; --i) {
            if (bits[i] == 0) {
                break;
            }
            ++count;
        }
        return count % 2 == 0;

    }

    // 718. 最长重复子数组
    public int findLength(final int[] A, final int[] B) {
        final int[] dp = new int[B.length + 1];
        int res = 0;
        for (int i = 0; i < A.length; ++i) {
            for (int j = B.length - 1; j >= 0; --j) {
                if (A[i] == B[j]) {
                    dp[j + 1] = dp[j] + 1;
                    if (res < dp[j + 1]) {
                        res = dp[j + 1];
                    }
                } else {
                    dp[j + 1] = 0;
                }
            }

        }
        return res;

    }

    // 718. 最长重复子数组
    public int findLength2(final int[] A, final int[] B) {
        int m = A.length;
        int n = B.length;
        int res = 0;
        for (int i = 0; i < m; ++i) {
            int length = Math.min(m - i, n);
            int cur = findMaxLength(A, B, i, 0, length);
            res = Math.max(cur, res);
        }
        for (int i = 0; i < n; ++i) {
            int length = Math.min(n - i, m);
            int cur = findMaxLength(A, B, 0, i, length);
            res = Math.max(cur, res);
        }
        return res;

    }

    private int findMaxLength(int[] A, int[] B, int startA, int startB, int length) {
        int count = 0;
        int cur = 0;
        int i = 0;
        while (i < length) {
            if (A[startA + i] == B[startB + i]) {
                ++cur;
            } else {
                count = Math.max(count, cur);
                cur = 0;
            }
            ++i;
        }
        return Math.max(cur, count);
    }

    // 724. 寻找数组的中心下标
    public int pivotIndex(final int[] nums) {
        int leftSum = 0;
        int sum = Arrays.stream(nums).sum();
        for (int i = 0; i < nums.length; ++i) {
            if (leftSum == sum - nums[i] - leftSum) {
                return i;
            }
            leftSum += nums[i];
        }
        return -1;

    }

    // 729. 我的日程安排表 I
    class MyCalendar {
        List<int[]> list;

        public MyCalendar() {
            list = new ArrayList<>();
        }

        public boolean book(int start, int end) {
            for (int[] cur : list) {
                if (end > cur[0] && start < cur[1]) {
                    return false;
                }
            }
            list.add(new int[] { start, end });
            return true;
        }
    }

    // 746. 使用最小花费爬楼梯
    public int minCostClimbingStairs(final int[] cost) {
        int pre = cost[0];
        int cur = cost[1];
        for (int i = 2; i < cost.length; ++i) {
            int next = cost[i] + Math.min(pre, cur);
            pre = cur;
            cur = next;
        }
        return Math.min(pre, cur);

    }

    // 747. 至少是其他数字两倍的最大数
    public int dominantIndex(final int[] nums) {
        int index = 0;
        int max = nums[0];
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] > max) {
                index = i;
                max = nums[i];
            }
        }
        for (int i = 0; i < nums.length; ++i) {
            if (i != index && max < nums[i] * 2) {
                return -1;
            }
        }
        return index;

    }

    // 766. 托普利茨矩阵
    public boolean isToeplitzMatrix(final int[][] matrix) {
        for (int i = 1; i < matrix.length; ++i) {
            for (int j = 1; j < matrix[0].length; ++j) {
                if (matrix[i][j] != matrix[i - 1][j - 1]) {
                    return false;
                }
            }
        }
        return true;

    }

    // 769. 最多能完成排序的块
    public int maxChunksToSorted(final int[] arr) {
        int res = 0;
        int max = arr[0];
        for (int i = 0; i < arr.length; ++i) {
            max = Math.max(max, arr[i]);
            if (i == max) {
                ++res;
            }
        }
        return res;

    }

    // 775. 全局倒置与局部倒置
    public boolean isIdealPermutation(final int[] A) {
        for (int i = 0; i < A.length; ++i) {
            for (int j = i + 2; j < A.length; ++j) {
                if (A[i] > A[j]) {
                    return false;
                }
            }
        }
        return true;
    }

    // 775. 全局倒置与局部倒置
    public boolean isIdealPermutation2(final int[] A) {
        int min = A[A.length - 1];
        for (int i = A.length - 1; i >= 2; --i) {
            min = Math.min(min, A[i]);
            if (A[i - 2] > min) {
                return false;
            }
        }
        return true;

    }

    // 1779. 找到最近的有相同 X 或 Y 坐标的点
    public int nearestValidPoint(int x, int y, int[][] points) {
        int distance = Integer.MAX_VALUE;
        int index = -1;
        for (int i = 0; i < points.length; ++i) {
            if (points[i][0] == x || points[i][1] == y) {
                int curDistance = Math.abs(x - points[i][0]) + Math.abs(y - points[i][1]);
                if (curDistance < distance) {
                    distance = curDistance;
                    index = i;
                }
            }
        }
        return index;

    }

    // 792. 匹配子序列的单词数--超时
    public int numMatchingSubseq(final String S, final String[] words) {
        int count = 0;
        for (String word : words) {
            int i = 0;
            for (char s : S.toCharArray()) {
                if (s == word.charAt(i)) {
                    ++i;
                }
                if (i == word.length()) {
                    ++count;
                    break;
                }
            }

        }
        return count;

    }

    // 792. 匹配子序列的单词数
    public int numMatchingSubseq2(final String S, final String[] words) {
        int count = 0;
        final ArrayList<Node>[] heads = new ArrayList[26];
        for (int i = 0; i < 26; ++i) {
            heads[i] = new ArrayList<>();
        }
        for (final String word : words) {
            heads[word.charAt(0) - 'a'].add(new Node(word, 0));
        }
        for (final char a : S.toCharArray()) {
            final ArrayList<Node> old = heads[a - 'a'];
            heads[a - 'a'] = new ArrayList<>();
            for (final Node node : old) {
                ++node.index;
                if (node.index == node.word.length()) {
                    ++count;
                } else {
                    heads[node.word.charAt(node.index) - 'a'].add(node);
                }
            }
            old.clear();
        }
        return count;

    }

    public class Node {
        int index;
        String word;

        public Node(final String word, final int index) {
            this.index = index;
            this.word = word;
        }

    }

    // 795. 区间子数组个数
    public int numSubarrayBoundedMax(final int[] A, final int L, final int R) {
        return count(A, R) - count(A, L - 1);
    }

    private int count(int[] A, int bound) {
        int count = 0;
        int res = 0;
        for (int i = 0; i < A.length; ++i) {
            if (A[i] <= bound) {
                ++count;
                res += count;
            } else {
                count = 0;
            }
        }
        return res;
    }

    // 825. 适龄的朋友
    public int numFriendRequests(final int[] ages) {
        int res = 0;
        int[] count = new int[121];
        for (int age : ages) {
            ++count[age];
        }
        for (int ageA = 1; ageA < count.length; ++ageA) {
            for (int ageB = 1; ageB < count.length; ++ageB) {
                if (ageB > 0.5 * ageA + 7 && ageB <= ageA && ((ageB <= 100)) || ((ageA >= 100))) {
                    res += count[ageA] * count[ageB];
                    if (ageA == ageB) {
                        res -= count[ageA];
                    }
                }
            }
        }
        return res;

    }

    // 830. 较大分组的位置
    public List<List<Integer>> largeGroupPositions(final String S) {
        List<List<Integer>> res = new ArrayList<>();
        int left = 0;
        int right = 1;
        while (right < S.length()) {
            if (S.charAt(right) != S.charAt(right - 1)) {
                if (right - left >= 3) {
                    res.add(Arrays.asList(left, right - 1));
                }
                left = right;
            }
            ++right;
        }
        if (right - left >= 3) {
            res.add(Arrays.asList(left, right - 1));
        }
        return res;

    }

    // 830. 较大分组的位置
    public List<List<Integer>> largeGroupPositions2(final String S) {
        final List<List<Integer>> list = new ArrayList<>();
        int i = 0;
        for (int j = 0; j < S.length(); ++j) {
            if (j == S.length() - 1 || S.charAt(j) != S.charAt(j + 1)) {
                if (j - i + 1 >= 3) {
                    list.add(Arrays.asList(i, j));
                }
                i = j + 1;
            }
        }
        return list;

    }

    // 832. 翻转图像
    public int[][] flipAndInvertImage(final int[][] A) {
        for (int[] row : A) {
            flipAndInvert(row);
        }
        return A;

    }

    private void flipAndInvert(int[] row) {
        int left = 0;
        int right = row.length - 1;
        while (left < right) {
            int temp = row[left];
            row[left] = row[right];
            row[right] = temp;
            ++left;
            --right;
        }
        for (int i = 0; i < row.length; ++i) {
            row[i] = row[i] == 0 ? 1 : 0;
        }
    }

    // 840. 矩阵中的幻方
    public int numMagicSquaresInside(final int[][] grid) {
        int count = 0;
        for (int i = 1; i < grid.length - 1; ++i) {
            for (int j = 1; j < grid[0].length - 1; ++j) {
                if (grid[i][j] != 5) {
                    continue;
                }
                if (magic(grid[i - 1][j - 1], grid[i - 1][j], grid[i - 1][j + 1], grid[i][j - 1], grid[i][j],
                        grid[i][j + 1], grid[i + 1][j - 1], grid[i + 1][j], grid[i + 1][j + 1])) {
                    ++count;
                }
            }
        }
        return count;
    }

    private boolean magic(int... nums) {
        int[] counts = new int[16];
        for (int num : nums) {
            ++counts[num];
        }
        for (int i = 1; i <= 9; ++i) {
            if (counts[i] != 1) {
                return false;
            }
        }
        return (nums[0] + nums[1] + nums[2] == 15) && (nums[3] + nums[4] + nums[5] == 15)
                && (nums[6] + nums[7] + nums[8] == 15) && (nums[0] + nums[3] + nums[6] == 15)
                && (nums[1] + nums[4] + nums[7] == 15) && (nums[2] + nums[5] + nums[8] == 15)
                && (nums[0] + nums[4] + nums[8] == 15) && (nums[2] + nums[4] + nums[6] == 15);
    }

    // 面试题 01.01. 判定字符是否唯一
    public boolean isUnique(final String astr) {
        final Set<Character> set = new HashSet<>();
        for (final char c : astr.toCharArray()) {
            if (!set.contains(c)) {
                set.add(c);
            } else {
                return false;
            }
        }
        return true;

    }

    // 面试题 01.02. 判定是否互为字符重排
    public boolean CheckPermutation(final String s1, final String s2) {
        final char[] a1 = s1.toCharArray();
        final char[] a2 = s2.toCharArray();
        Arrays.sort(a1);
        Arrays.sort(a2);
        return Arrays.equals(a1, a2);

    }

    // 面试题 01.02. 判定是否互为字符重排
    public boolean CheckPermutation2(final String s1, final String s2) {
        int[] counts = new int[26];
        for (char c : s1.toCharArray()) {
            ++counts[c - 'a'];
        }
        for (char c : s2.toCharArray()) {
            --counts[c - 'a'];
        }
        for (int count : counts) {
            if (count != 0) {
                return false;
            }
        }
        return true;

    }

    // 面试题 01.01. 判定字符是否唯一
    public boolean isUnique2(final String astr) {
        for (int i = 0; i < astr.length() - 1; ++i) {
            if (astr.indexOf(astr.charAt(i), i + 1) != -1) {
                return false;
            }
        }
        return true;
    }

    // 面试题 01.01. 判定字符是否唯一
    public boolean isUnique3(final String astr) {
        int[] counts = new int[26];
        for (char c : astr.toCharArray()) {
            ++counts[c - 'a'];
        }
        for (int count : counts) {
            if (count > 1) {
                return false;
            }
        }
        return true;
    }

    /**
     * 输入："Mr John Smith ", 13 输出："Mr%20John%20Smith"
     * 
     * @param S
     * @param length
     * @return
     */
    public String replaceSpaces(final String S, final int length) {
        final StringBuilder builder = new StringBuilder();
        for (int i = 0; i < length; ++i) {
            if (S.charAt(i) == ' ') {
                builder.append("%20");
            } else {
                builder.append(S.charAt(i));
            }
        }
        return builder.toString();
    }

    public boolean canPermutePalindrome(final String s) {
        final Map<Character, Integer> map = new HashMap<>();
        for (final char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        int oddCount = 0;
        for (final int num : map.values()) {
            if (num % 2 == 1) {
                ++oddCount;
            }
        }
        return oddCount <= 1;

    }

    // public boolean oneEditAway(String first, String second) {
    // if (first.length() == second.length()) {
    // int count = 0;
    // for (int i = 0; i < first.length(); ++i) {
    // if (first.charAt(i) != second.charAt(i)) {
    // ++count;
    // if (count > 1) {
    // return false;
    // }
    // }
    // }
    // return true;
    // } else if (Math.abs(first.length() - second.length()) == 1) {
    // if (second.length() > first.length()) {
    // String tmp;
    // tmp = first;
    // first = second;
    // second = tmp;
    // }
    // for (int i = 0; i < first.length(); ++i) {
    // if (first.charAt(i) != second.charAt(i)) {

    // }

    // }

    // }

    // }
    public String compressString(final String S) {
        if (S.isEmpty()) {
            return S;
        }
        final StringBuilder builder = new StringBuilder();
        builder.append(S.charAt(0));
        int count = 1;
        for (int i = 1; i < S.length(); ++i) {
            if (builder.charAt(builder.length() - 1) == S.charAt(i)) {
                ++count;
            } else {
                builder.append(count);
                count = 1;
                builder.append(S.charAt(i));
            }
        }
        builder.append(count);
        return builder.length() >= S.length() ? S : builder.toString();
    }

    public boolean isFlipedString(final String s1, final String s2) {
        if (s1.equals(s2)) {
            return true;
        }
        if (s1.length() != s2.length()) {
            return false;
        }
        final char[] chars1 = s1.toCharArray();
        final char[] chars2 = s2.toCharArray();
        reverse2(chars1, 0, chars1.length - 1);
        for (int i = 0; i < chars1.length; ++i) {
            final char[] chars1Clone = chars1.clone();
            reverse2(chars1Clone, 0, i);
            reverse2(chars1Clone, i + 1, chars1Clone.length - 1);
            if (Arrays.equals(chars1Clone, chars2)) {
                return true;
            }
        }
        return false;
    }

    private void reverse2(final char[] chars1, int left, int right) {
        char temp;
        while (left < right) {
            temp = chars1[left];
            chars1[left] = chars1[right];
            chars1[right] = temp;
            ++left;
            --right;
        }
    }

    public boolean isFlipedString2(final String s1, final String s2) {
        final String s = s1 + s1;
        return s.contains(s2) && (s.length() == s2.length() * 2);

    }

    // 849. 到最近的人的最大距离
    public int maxDistToClosest(final int[] seats) {
        int max = 0;
        int i = 0;
        while (i < seats.length) {
            if (seats[i] == 1) {
                break;
            }
            ++i;
        }
        int j = seats.length - 1;
        while (j >= 0) {
            if (seats[j] == 1) {
                break;
            }
            --j;
        }
        max = Math.max(i, seats.length - 1 - j);
        int count = 0;
        int cur = 0;
        while (i < j) {
            if (seats[i] == 1) {
                count = Math.max(count, cur);
                cur = 0;
            } else {
                ++cur;
            }
            ++i;
        }
        count = Math.max(count, cur);
        return Math.max(max, (count + 1) / 2);

    }

    // 867. 转置矩阵
    public int[][] transpose(final int[][] A) {
        int[][] res = new int[A[0].length][A.length];
        for (int i = 0; i < A.length; ++i) {
            for (int j = 0; j < A[0].length; ++j) {
                res[j][i] = A[i][j];
            }
        }
        return res;

    }

    // 870. 优势洗牌
    public static int[] advantageCount(final int[] A, final int[] B) {
        boolean[] visited = new boolean[A.length];
        Arrays.sort(A);
        for (int i = 0; i < B.length; ++i) {
            boolean flag = false;
            for (int j = 0; j < A.length; ++j) {
                if (!visited[j] && A[j] > B[i]) {
                    visited[j] = true;
                    flag = true;
                    B[i] = A[j];
                    break;
                }
            }
            if (!flag) {
                for (int j = 0; j < A.length; ++j) {
                    if (!visited[j]) {
                        visited[j] = true;
                        B[i] = A[j];
                        break;
                    }
                }

            }
        }
        return B;

    }

    // 873. 最长的斐波那契子序列的长度
    public int lenLongestFibSubseq(final int[] A) {
        Set<Integer> set = new HashSet<>();
        for (int num : A) {
            set.add(num);
        }

        int max = 0;
        for (int i = 0; i < A.length; ++i) {
            for (int j = i + 1; j < A.length; ++j) {
                int f1 = A[i];
                int f2 = A[j];
                int count = 0;
                while (set.contains(f1 + f2)) {
                    int temp = f1 + f2;
                    f1 = f2;
                    f2 = temp;
                    ++count;
                }
                max = Math.max(max, count == 0 ? 0 : count + 2);
            }
        }
        return max;

    }

    // 888. 公平的糖果棒交换
    public int[] fairCandySwap(final int[] A, final int[] B) {
        int diff = (Arrays.stream(A).sum() - Arrays.stream(B).sum()) / 2;
        Set<Integer> set = new HashSet<>();
        for (int num : A) {
            set.add(num);
        }
        for (int i = 0; i < B.length; ++i) {
            if (set.contains(diff + B[i])) {
                return new int[] { diff + B[i], B[i] };
            }
        }
        return null;

    }

    // 896. 单调数列
    public boolean isMonotonic(final int[] A) {
        return isIncrese(A) || isDecrease(A);

    }

    private boolean isIncrese(int[] A) {
        for (int i = 1; i < A.length; ++i) {
            if (A[i - 1] > A[i]) {
                return false;
            }
        }
        return true;

    }

    private boolean isDecrease(int[] A) {
        for (int i = 1; i < A.length; ++i) {
            if (A[i - 1] < A[i]) {
                return false;
            }
        }
        return true;

    }

    // 896. 单调数列
    public boolean isMonotonic2(final int[] A) {
        int store = 0;
        for (int i = 0; i < A.length - 1; ++i) {
            final int c = Integer.compare(A[i], A[i + 1]);
            if (c != 0) {
                if (store != 0 && c != store) {
                    return false;
                }
                store = c;
            }
        }
        return true;

    }

    // 905. 按奇偶排序数组
    public int[] sortArrayByParity(final int[] A) {
        int left = 0;
        int right = A.length - 1;
        while (left < right) {
            while (left < right && A[left] % 2 == 0) {
                ++left;
            }
            while (left < right && A[right] % 2 == 1) {
                --right;
            }
            int temp = A[left];
            A[left] = A[right];
            A[right] = temp;
            ++left;
            --right;
        }
        return A;

    }

    // 922. 按奇偶排序数组 II
    public int[] sortArrayByParityII(final int[] A) {
        int i = 0;
        int j = 1;
        while (i < A.length && j < A.length) {
            while (i < A.length && A[i] % 2 == 0) {
                i += 2;
            }
            while (j < A.length && A[j] % 2 == 1) {
                j += 2;
            }
            if (i >= A.length || j >= A.length) {
                break;
            }
            int temp = A[i];
            A[i] = A[j];
            A[j] = temp;
            i += 2;
            j += 2;
        }
        return A;

    }

    // 941. 有效的山脉数组
    public boolean validMountainArray(final int[] A) {
        if (A == null || A.length < 3) {
            return false;
        }
        if (A[1] <= A[0] || A[A.length - 1] >= A[A.length - 2]) {
            return false;
        }
        boolean shouldUpFlag = true;
        for (int i = 0; i < A.length - 1; ++i) {
            if (shouldUpFlag) {
                if (A[i + 1] > A[i]) {
                    if (i == A.length - 2) {
                        return false;
                    }
                } else {
                    shouldUpFlag = false;
                }
            } else {
                if (A[i + 1] >= A[i]) {
                    return false;
                }
            }

        }
        return true;

    }

    // 941. 有效的山脉数组
    public boolean validMountainArray2(final int[] A) {
        int i = 0;
        while (i < A.length - 1 && A[i] < A[i + 1]) {
            ++i;
        }
        if (i == 0 || i == A.length - 1) {
            return false;
        }
        while (i < A.length - 1 && A[i] > A[i + 1]) {
            ++i;
        }
        return i == A.length - 1;

    }

    // 941. 有效的山脉数组
    public boolean validMountainArray3(final int[] A) {
        int i = 0;
        while (i < A.length) {
            if (A[i - 1] >= A[i]) {
                break;
            }
            ++i;
        }
        if (i == 1 || i == A.length) {
            return false;
        }
        while (i < A.length) {
            if (A[i - 1] <= A[i]) {
                return false;
            }
            ++i;
        }
        return i == A.length;

    }

    // 915. 分割数组
    public int partitionDisjoint(final int[] A) {
        int[] leftMax = new int[A.length];
        int[] rightMin = new int[A.length];
        leftMax[0] = A[0];
        for (int i = 1; i < A.length; ++i) {
            leftMax[i] = Math.max(A[i], leftMax[i - 1]);
        }
        rightMin[A.length - 1] = A[A.length - 1];
        for (int i = A.length - 2; i >= 0; --i) {
            rightMin[i] = Math.min(A[i], rightMin[i + 1]);
        }
        for (int i = 0; i < A.length - 1; ++i) {
            if (leftMax[i] <= rightMin[i + 1]) {
                return i + 1;
            }
        }
        return -1;

    }

    public int sumSubarrayMins(final int[] A) {
        final int MOD = 1_000_000_007;
        final Stack<RepInteger> stack = new Stack<>();
        int ans = 0;
        int dot = 0;
        for (int j = 0; j < A.length; ++j) {
            int count = 1;
            while (!stack.isEmpty() && stack.peek().val >= A[j]) {
                final RepInteger repInteger = stack.pop();
                count += repInteger.count;
                dot -= repInteger.count * repInteger.val;
            }
            stack.push(new RepInteger(A[j], count));
            dot += A[j] * count;
            ans += dot;
            ans %= MOD;
        }
        return ans;

    }

    public class RepInteger {
        int val;
        int count;

        public RepInteger(final int val, final int count) {
            this.val = val;
            this.count = count;
        }

    }

    public boolean hasGroupsSizeX(final int[] deck) {
        int N = deck.length;
        int[] count = new int[10000];
        for (int i = 0; i < deck.length; ++i) {
            ++count[deck[i]];
        }
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < count.length; ++i) {
            if (count[i] > 0) {
                list.add(count[i]);
            }
        }
        search: for (int X = 2; X <= N; ++X) {
            if (N % X == 0) {
                for (int num : list) {
                    if (num % X != 0) {
                        continue search;
                    }
                }
                return true;
            }
        }
        return false;

    }

    public int minFlipsMonoIncr(final String S) {
        int numOfOne = 0;
        final int[] dp = new int[S.length()];
        for (int i = 0; i < S.length(); ++i) {
            if (S.charAt(i) == '1') {
                dp[i] = i == 0 ? 0 : Math.min(dp[i - 1], numOfOne + 1);
                ++numOfOne;
            } else {
                dp[i] = i == 0 ? 0 : Math.min(dp[i - 1] + 1, numOfOne);

            }
        }
        return dp[S.length() - 1];

    }

    // 945. 使数组唯一的最小增量
    public int minIncrementForUnique(final int[] A) {
        int[] counts = new int[80001];
        for (int num : A) {
            ++counts[num];
        }
        int taken = 0;
        int result = 0;
        for (int i = 0; i < counts.length; ++i) {
            if (counts[i] > 1) {
                taken += counts[i] - 1;
                result -= (counts[i] - 1) * i;
            } else if (counts[i] == 0 && taken > 0) {
                --taken;
                result += i;
            }
        }
        return result;

    }

    public int[] deckRevealedIncreasing(final int[] deck) {
        if (deck == null || deck.length <= 1) {
            return deck;
        }
        Arrays.sort(deck);
        final Queue<Integer> queue = new LinkedList<>();
        for (int i = deck.length - 1; i >= 0; --i) {
            queue.add(deck[i]);
            if (i == 0) {
                break;
            }
            queue.add(queue.poll());
        }
        for (int i = deck.length - 1; i >= 0; --i) {
            deck[i] = queue.poll();
        }
        return deck;
    }

    // 954. 二倍数对数组
    public boolean canReorderDoubled(final int[] A) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : A) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        Integer[] B = new Integer[A.length];
        for (int i = 0; i < A.length; ++i) {
            B[i] = A[i];
        }
        Arrays.sort(B, new Comparator<Integer>() {
            @Override
            public int compare(Integer o1, Integer o2) {
                return Math.abs(o1) - Math.abs(o2);
            }
        });

        for (int i = 0; i < B.length; ++i) {
            if (map.getOrDefault(B[i], 0) == 0) {
                continue;
            }
            if (map.getOrDefault(B[i] * 2, 0) == 0) {
                return false;
            }
            map.put(B[i], map.getOrDefault(B[i], 0) - 1);
            map.put(B[i] * 2, map.getOrDefault(B[i] * 2, 0) - 1);
        }
        return true;

    }

    // 962. 最大宽度坡
    public int maxWidthRamp(final int[] A) {
        int max = 0;
        for (int i = A.length - 1; i >= max; --i) {
            for (int j = 0; j < i - max; ++j) {
                if (A[j] <= A[i]) {
                    max = Math.max(max, i - j);

                }
            }
        }
        return max;

    }

    // 977. 有序数组的平方
    public int[] sortedSquares(final int[] A) {
        for (int i = 0; i < A.length; ++i) {
            A[i] *= A[i];
        }
        Arrays.sort(A);
        return A;

    }

    // 977. 有序数组的平方
    public int[] sortedSquares2(final int[] A) {
        int i = 0;
        while (i < A.length) {
            if (A[i] >= 0) {
                break;
            }
            ++i;
        }
        int j = i - 1;
        int[] res = new int[A.length];
        int index = 0;
        while (j >= 0 && i < A.length) {
            if (A[j] * A[j] > A[i] * A[i]) {
                res[index++] = A[i] * A[i];
                ++i;
            } else {
                res[index++] = A[j] * A[j];
                j--;
            }
        }
        while (j >= 0) {
            res[index++] = A[j] * A[j];
            --j;
        }
        while (i < A.length) {
            res[index++] = A[i] * A[i];
            ++i;
        }
        return res;

    }

    public int majorityElement5(final int[] nums) {
        if (nums == null || nums.length == 0) {
            return -1;
        }
        int res = nums[0];
        int count = 1;
        for (int i = 1; i < nums.length; ++i) {
            if (res == nums[i]) {
                ++count;
            } else {
                if (count > 0) {
                    --count;
                } else {
                    res = nums[i];
                    count = 1;
                }
            }
        }

        count = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == res) {
                ++count;
            }
        }
        return count * 2 > nums.length ? res : -1;

    }

    // 509. 斐波那契数 // 剑指 Offer 10- I. 斐波那契数列
    public int fib(final int N) {
        if (N == 0) {
            return 0;
        }
        int f1 = 0;
        int f2 = 1;
        for (int i = 1; i < N; ++i) {
            int temp = f1 + f2;
            f1 = f2;
            f2 = temp;
        }
        return f2;

    }

    // 989. 数组形式的整数加法
    public List<Integer> addToArrayForm(int[] A, int K) {
        List<Integer> list = new ArrayList<>();
        int i = A.length - 1;
        while (i >= 0 || K > 0) {
            if (i >= 0) {
                K += A[i];
            }
            list.add(K % 10);
            K /= 10;
            --i;
        }
        Collections.reverse(list);
        return list;

    }

    // 1002. 查找常用字符
    public List<String> commonChars(String[] A) {
        int[] counts = new int[26];
        for (char a : A[0].toCharArray()) {
            ++counts[a - 'a'];
        }
        for (String s : A) {
            int[] curCount = new int[26];
            for (char a : s.toCharArray()) {
                ++curCount[a - 'a'];
            }
            for (int i = 0; i < counts.length; ++i) {
                counts[i] = Math.min(counts[i], curCount[i]);
            }
        }
        List<String> list = new ArrayList<>();
        for (int i = 0; i < counts.length; ++i) {
            for (int j = 0; j < counts[i]; ++j) {
                list.add(String.valueOf((char) (i + 'a')));
            }
        }
        return list;

    }

    // 4
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int[] result = new int[nums1.length + nums2.length];
        int i = 0;
        int j = 0;
        int index = 0;
        while (i < nums1.length && j < nums2.length) {
            if (nums1[i] < nums2[j]) {
                result[index++] = nums1[i++];
            } else {
                result[index++] = nums2[j++];
            }
        }
        while (i < nums1.length) {
            result[index++] = nums1[i++];
        }
        while (j < nums2.length) {
            result[index++] = nums2[j++];
        }
        if ((nums1.length + nums2.length) % 2 == 1) {
            return result[(nums1.length + nums2.length) / 2];
        }
        return ((double) (result[(nums1.length + nums2.length) / 2] + result[(nums1.length + nums2.length - 1) / 2]))
                / 2;

    }

    // 41. 缺失的第一个正数
    public int firstMissingPositive(int[] nums) {
        for (int i = 0; i < nums.length; ++i) {
            while (nums[i] >= 1 && nums[i] <= nums.length && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[i];
                nums[i] = nums[temp - 1];
                nums[temp - 1] = temp;
            }
        }
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return nums.length + 1;

    }

    // 1018. 可被 5 整除的二进制前缀
    public List<Boolean> prefixesDivBy5(final int[] A) {
        List<Boolean> result = new ArrayList<>();
        int num = 0;
        for (int i = 0; i < A.length; ++i) {
            num = ((num << 1) + A[i]) % 5;
            result.add(num == 0);
        }
        return result;

    }

    public int game(final int[] guess, final int[] answer) {
        int count = 0;
        for (int i = 0; i < guess.length; ++i) {
            count += guess[i] == answer[i] ? 1 : 0;
        }
        return count;

    }

    public int numRookCaptures(final char[][] board) {
        int count = 0;
        int i = 0;
        int j = 0;
        out: for (i = 0; i < board.length; ++i) {
            for (j = 0; j < board[0].length; ++j) {
                if (board[i][j] == 'R') {
                    break out;
                }
            }
        }
        final int posC = j;
        final int posR = i;
        // 从R的位置往上查找
        while (--i >= 0) {
            if (board[i][j] == 'B') {
                break;
            } else if (board[i][j] == 'p') {
                ++count;
                break;
            }
        }
        i = posR;
        // 从R的位置往下查找
        while (++i < board.length) {
            if (board[i][j] == 'B') {
                break;
            } else if (board[i][j] == 'p') {
                ++count;
                break;
            }
        }
        i = posR;
        // 从R的位置往左查找
        while (--j >= 0) {
            if (board[i][j] == 'B') {
                break;
            } else if (board[i][j] == 'p') {
                ++count;
                break;
            }
        }
        j = posC;
        // 从R的位置往右查找
        while (++j < board[0].length) {
            if (board[i][j] == 'B') {
                break;
            } else if (board[i][j] == 'p') {
                ++count;
                break;
            }
        }
        return count;

    }

    // 985. 查询后的偶数和
    public int[] sumEvenAfterQueries(final int[] A, final int[][] queries) {
        int sum = 0;
        int[] answer = new int[queries.length];
        for (int i = 0; i < A.length; ++i) {
            if (A[i] % 2 == 0) {
                sum += A[i];
            }
        }
        for (int i = 0; i < queries.length; ++i) {
            int val = queries[i][0];
            int index = queries[i][1];
            if (A[index] % 2 == 0) {
                sum -= A[index];
            }
            A[index] += val;
            if (A[index] % 2 == 0) {
                sum += A[index];
            }
            answer[i] = sum;
        }
        return answer;

    }

    public int maxTurbulenceSize(final int[] A) {
        int result = 1;
        int count = 1;
        int pre = 0;
        boolean flag = false;
        for (int i = 0; i < A.length - 1; ++i) {
            final int sign = Integer.compare(A[i], A[i + 1]);
            if (sign == 0) {
                result = Math.max(result, count);
                count = 1;
                flag = false;
                pre = 0;
            } else if (!flag) {
                flag = true;
                pre = sign;
                ++count;
            } else {
                if (-sign == pre) {
                    ++count;
                    pre = sign;
                } else {
                    result = Math.max(result, count);
                    count = 2;
                }
            }
        }
        result = Math.max(result, count);
        return result;

    }

    public int maxTurbulenceSize2(final int[] A) {
        int ret = 1;
        int left = 0;
        int right = 0;
        while (right < A.length - 1) {
            if (left == right) {
                if (A[left] == A[left + 1]) {
                    ++left;
                }
                ++right;
            } else {
                if (A[right - 1] < A[right] && A[right] > A[right + 1]) {
                    ++right;
                } else if (A[right - 1] > A[right] && A[right] < A[right + 1]) {
                    ++right;
                } else {
                    left = right;
                }
            }
            ret = Math.max(ret, right - left + 1);
        }
        return ret;

    }

    // 1464. 数组中两元素的最大乘积
    public int maxProduct1464(int[] nums) {
        int max1 = Integer.MIN_VALUE;
        int max2 = Integer.MIN_VALUE;
        for (int num : nums) {
            if (num >= max1) {
                max2 = max1;
                max1 = num;
            } else if (num >= max2) {
                max2 = num;
            }
        }
        return (max1 - 1) * (max2 - 1);

    }

    // 1470. 重新排列数组
    public int[] shuffle(int[] nums, int n) {
        int[] res = new int[nums.length];
        int i = 0;
        int j = n;
        int index = 0;
        while (i < n) {
            res[index++] = nums[i++];
            res[index++] = nums[j++];
        }
        return res;

    }

    // 1486. 数组异或操作
    public int xorOperation(int n, int start) {
        int res = 0;
        for (int i = 0; i < n; ++i) {
            res ^= start + 2 * i;
        }
        return res;

    }

    // 1480. 一维数组的动态和
    public int[] runningSum(int[] nums) {
        for (int i = 1; i < nums.length; ++i) {
            nums[i] += nums[i - 1];
        }
        return nums;

    }

    // 1010. 总持续时间可被 60 整除的歌曲--超时
    public int numPairsDivisibleBy60(final int[] time) {
        int count = 0;
        for (int i = 0; i < time.length; ++i) {
            for (int j = i + 1; j < time.length; ++j) {
                if ((time[i] + time[j]) % 60 == 0) {
                    ++count;
                }
            }
        }
        return count;
    }

    // 1010. 总持续时间可被 60 整除的歌曲
    public int numPairsDivisibleBy60_2(final int[] time) {
        int[] counts = new int[60];
        for (int i = 0; i < time.length; ++i) {
            time[i] %= 60;
            ++counts[time[i]];
        }
        int count = 0;
        for (int i = 0; i <= 30; ++i) {
            if (i == 0 || i == 30) {
                count += counts[i] * (counts[i] - 1) / 2;
            } else {
                count += counts[i] * counts[60 - i];
            }
        }
        return count;

    }

    public int subarraysDivByK(final int[] A, final int K) {
        int[] dp = new int[A.length + 1];

        for (int i = 0; i < A.length; ++i) {
            dp[i + 1] = A[i] + dp[i];
        }

        int[] count = new int[K];
        for (int i = 0; i < dp.length; ++i) {
            ++count[(dp[i] % K + K) % K];
        }
        int res = 0;
        for (int i = 0; i < count.length; ++i) {
            res += count[i] * (count[i] - 1) / 2;
        }
        return res;

    }

    // 1014. 最佳观光组合
    public int maxScoreSightseeingPair(final int[] A) {
        int max = 0;
        int pre = A[0] + 0;
        for (int i = 1; i < A.length; ++i) {
            max = Math.max(max, pre + A[i] - i);
            pre = Math.max(pre, A[i] + i);
        }
        return max;

    }

    // 1011. 在 D 天内送达包裹的能力
    public int shipWithinDays(final int[] weights, final int D) {
        int left = Arrays.stream(weights).max().getAsInt();
        int right = Arrays.stream(weights).sum();
        int ans = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (canShip(weights, mid, D)) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;

    }

    private boolean canShip(int[] weights, int capacity, int D) {
        int days = 1;
        int cur = capacity;
        for (int i = 0; i < weights.length; ++i) {
            if (weights[i] > cur) {
                ++days;
                cur = capacity;
            }
            cur -= weights[i];
        }
        return days <= D;
    }

    // 1031. 两个非重叠子数组的最大和
    public int maxSumTwoNoOverlap(final int[] A, final int L, final int M) {
        int[][] dp = new int[A.length][4];
        // 从左边 L元素
        int curMax = 0;
        for (int i = 0; i < L; ++i) {
            curMax += A[i];
        }
        dp[L - 1][0] = curMax;
        for (int i = L; i < A.length; ++i) {
            curMax += A[i] - A[i - L];
            dp[i][0] = Math.max(dp[i - 1][0], curMax);
        }
        // 从左边 M元素
        curMax = 0;
        for (int i = 0; i < M; ++i) {
            curMax += A[i];
        }
        dp[M - 1][1] = curMax;
        for (int i = M; i < A.length; ++i) {
            curMax += A[i] - A[i - M];
            dp[i][1] = Math.max(curMax, dp[i - 1][1]);
        }
        // 从右边 L元素
        curMax = 0;
        for (int i = A.length - 1; i >= A.length - L; --i) {
            curMax += A[i];
        }
        dp[A.length - L][2] = curMax;
        for (int i = A.length - L - 1; i >= 0; --i) {
            curMax += A[i] - A[i + L];
            dp[i][2] = Math.max(dp[i + 1][2], curMax);
        }

        // 从右边 M元素
        curMax = 0;
        for (int i = A.length - 1; i >= A.length - M; --i) {
            curMax += A[i];
        }
        dp[A.length - M][3] = curMax;
        for (int i = A.length - M - 1; i >= 0; --i) {
            curMax += A[i] - A[i + M];
            dp[i][3] = Math.max(dp[i + 1][3], curMax);
        }

        int max = 0;
        for (int i = L - 1; i < A.length - M; ++i) {
            max = Math.max(max, dp[i][0] + dp[i + 1][3]);
        }
        for (int i = M - 1; i < A.length - L; ++i) {
            max = Math.max(max, dp[i][1] + dp[i + 1][2]);
        }
        return max;
    }

    // 1031. 两个非重叠子数组的最大和
    public int maxSumTwoNoOverlap2(final int[] A, final int L, final int M) {
        return Math.max(calculateDP(A, L, M), calculateDP(A, M, L));
    }

    private int calculateDP(final int[] A, final int L, final int M) {
        // final int[][] dp = new int[A.length][2];
        // int cursum = 0;
        // for (int i = 0; i < L; ++i) {
        // cursum += A[i];
        // }
        // dp[L - 1][0] = cursum;
        // for (int i = L; i < A.length; ++i) {
        // cursum += A[i] - A[i - L];
        // dp[i][0] = Math.max(cursum, dp[i - 1][0]);
        // }

        // cursum = 0;
        // for (int i = A.length - 1; i >= A.length - M; --i) {
        // cursum += A[i];
        // }
        // dp[A.length - M][1] = cursum;
        // for (int i = A.length - M - 1; i >= 0; --i) {
        // cursum += A[i] - A[i + M];
        // dp[i][1] = Math.max(cursum, dp[i - 1][1]);

        // }
        // int res = 0;
        // for (int i = L; i <= A.length - M; ++i) {
        // res = Math.max(res, dp[i - 1][0] + dp[i][1]);
        // }

        // return res;
        int[][] dp = new int[A.length][2];
        int curSum = 0;
        for (int i = 0; i < L; ++i) {
            curSum += A[i];
        }
        dp[L - 1][0] = curSum;
        for (int i = L; i < A.length; ++i) {
            curSum += A[i] - A[i - L];
            dp[i][0] = Math.max(dp[i - 1][0], curSum);
        }

        curSum = 0;
        for (int i = A.length - 1; i >= A.length - M; --i) {
            curSum += A[i];
        }
        dp[A.length - M][1] = curSum;
        for (int i = A.length - M - 1; i >= 0; --i) {
            curSum += A[i] - A[i + M];
            dp[i][1] = Math.max(dp[i + 1][1], curSum);
        }
        int max = 0;
        for (int i = L - 1; i < A.length - M; ++i) {
            max = Math.max(max, dp[i][0] + dp[i + 1][1]);
        }
        return max;

    }

    public int maxUncrossedLines(final int[] A, final int[] B) {
        final int[][] dp = new int[A.length + 1][B.length + 1];
        for (int i = 0; i < A.length; ++i) {
            for (int j = 0; j < B.length; ++j) {
                if (A[i] == B[j]) {
                    dp[i + 1][j + 1] = dp[i][j] + 1;
                } else {
                    dp[i + 1][j + 1] = Math.max(dp[i][j + 1], dp[i + 1][j]);
                }
            }
        }
        return dp[A.length][B.length];
    }

    // 1160. 拼写单词
    public int countCharacters(final String[] words, final String chars) {
        int[] counts = new int[26];
        for (char a : chars.toCharArray()) {
            ++counts[a - 'a'];
        }
        int result = 0;
        for (String word : words) {
            int[] copyCounts = counts.clone();
            int count = 0;
            for (char c : word.toCharArray()) {
                if (copyCounts[c - 'a'] > 0) {
                    --copyCounts[c - 'a'];
                    ++count;
                } else {
                    break;
                }
            }
            if (count == word.length()) {
                result += count;
            }
        }
        return result;

    }

    // 1051. 高度检查器
    public int heightChecker(final int[] heights) {
        int[] copy = heights.clone();
        Arrays.sort(copy);
        int count = 0;
        for (int i = 0; i < heights.length; ++i) {
            if (copy[i] != heights[i]) {
                ++count;
            }
        }
        return count;

    }

    // 1051. 高度检查器
    public int heightChecker2(final int[] heights) {
        int[] counts = new int[101];
        for (int height : heights) {
            ++counts[height];
        }
        int count = 0;
        int index = 0;
        for (int i = 1; i < counts.length; ++i) {
            for (int j = 0; j < counts[i]; ++j) {
                if (heights[index++] != i) {
                    ++count;
                }
            }
        }
        return count;

    }

    // 1052. 爱生气的书店老板
    public int maxSatisfied(final int[] customers, final int[] grumpy, final int X) {
        int satisfy = 0;
        for (int i = 0; i < grumpy.length; ++i) {
            if (grumpy[i] == 0) {
                satisfy += customers[i];
            }
        }

        int calmDownSatisfy = 0;
        for (int i = 0; i < X; ++i) {
            if (grumpy[i] == 1) {
                calmDownSatisfy += customers[i];
            }
        }
        int maxCalmDownSatisfy = calmDownSatisfy;
        for (int i = X; i < grumpy.length; ++i) {
            if (grumpy[i] == 1) {
                calmDownSatisfy += customers[i];
            }
            if (grumpy[i - X] == 1) {
                calmDownSatisfy -= customers[i - X];
            }
            maxCalmDownSatisfy = Math.max(maxCalmDownSatisfy, calmDownSatisfy);
        }
        return maxCalmDownSatisfy + satisfy;

    }

    // 1053. 交换一次的先前排列
    public int[] prevPermOpt1(final int[] A) {
        int i = A.length - 2;
        while (i >= 0) {
            if (A[i] > A[i + 1]) {
                break;
            }
            --i;
        }
        if (i < 0) {
            return A;
        }
        int j = A.length - 1;
        while (j > i) {
            if (A[j] < A[i] && A[j] != A[j - 1]) {
                int temp = A[j];
                A[j] = A[i];
                A[i] = temp;
                return A;
            }
            --j;
        }
        return A;

    }

    // 1089. 复写零
    public void duplicateZeros(final int[] arr) {
        boolean flag = false;
        for (int i = 0; i < arr.length; ++i) {
            if (!flag && arr[i] == 0) {
                flag = true;
                continue;
            }
            if (flag) {
                flag = false;
                for (int j = arr.length - 1; j >= i; --j) {
                    arr[j] = arr[j - 1];
                }
            }
        }

    }

    // 1089. 复写零
    public void duplicateZeros2(final int[] arr) {
        int count = 0;
        int i = 0;
        while (i < arr.length) {
            if (arr[i] == 0) {
                count += 2;
            } else {
                count += 1;
            }
            if (count == arr.length || count == arr.length + 1) {
                break;
            }
            ++i;
        }
        int index = arr.length - 1;
        if (count == arr.length + 1) {
            --i;
            --index;
            arr[arr.length - 1] = 0;
        }
        while (index >= 0) {
            if (arr[i] == 0) {
                arr[index--] = arr[i];
            }
            arr[index--] = arr[i--];
        }

    }

    public int[] corpFlightBookings(final int[][] bookings, final int n) {
        final int[] res = new int[n];
        for (int i = 0; i < bookings.length; ++i) {
            for (int j = bookings[i][0]; j <= bookings[i][1]; ++j) {
                res[j - 1] += bookings[i][2];
            }
        }
        return res;
    }

    // 1109. 航班预订统计
    public int[] corpFlightBookings2(final int[][] bookings, final int n) {
        int[] res = new int[n];
        for (int[] booking : bookings) {
            res[booking[0] - 1] += booking[2];
            if (booking[1] < n) {
                res[booking[1]] -= booking[2];
            }
        }
        for (int i = 1; i < res.length; ++i) {
            res[i] += res[i - 1];
        }
        return res;

    }

    // 1122. 数组的相对排序
    public int[] relativeSortArray(final int[] arr1, final int[] arr2) {
        int p = 0;
        int tmp;
        for (int i = 0; i < arr2.length; ++i) {
            for (int j = p; j < arr1.length; ++j) {
                if (arr2[i] == arr1[j]) {
                    tmp = arr1[j];
                    arr1[j] = arr1[p];
                    arr1[p] = tmp;
                    ++p;
                }
            }
        }
        Arrays.sort(arr1, p, arr1.length);
        return arr1;

    }

    // 1122. 数组的相对排序
    public int[] relativeSortArray2(int[] arr1, int[] arr2) {
        int[] result = new int[arr1.length];
        int[] counts = new int[1001];
        for (int num : arr1) {
            ++counts[num];
        }
        int index = 0;
        for (int i = 0; i < arr2.length; ++i) {
            for (int j = 0; j < counts[arr2[i]]; ++j) {
                result[index++] = arr2[i];
            }
            counts[arr2[i]] = 0;
        }
        for (int i = 0; i < counts.length; ++i) {
            for (int j = 0; j < counts[i]; ++j) {
                result[index++] = i;
            }
        }
        return result;

    }

    // 1287. 有序数组中出现次数超过25%的元素
    public int findSpecialInteger(final int[] arr) {
        int span = arr.length / 4;
        for (int i = 0; i < arr.length; ++i) {
            if (arr[i] == arr[i + span]) {
                return arr[i];
            }
        }
        return -1;
    }

    public int[] replaceElements(final int[] arr) {
        int[] result = new int[arr.length];
        result[result.length - 1] = arr[arr.length - 1];
        for (int i = arr.length - 2; i >= 0; --i) {
            result[i] = Math.max(result[i + 1], arr[i + 1]);
        }
        result[result.length - 1] = -1;
        return result;

    }

    // 1313. 解压缩编码列表
    public int[] decompressRLElist(final int[] nums) {
        int count = 0;
        for (int i = 0; i < nums.length; i += 2) {
            count += nums[i];
        }
        int[] result = new int[count];
        int index = 0;
        for (int i = 1; i < nums.length; i += 2) {
            for (int j = 0; j < nums[i - 1]; ++j) {
                result[index++] = nums[i];
            }
        }
        return result;

    }

    // 剑指 Offer 03. 数组中重复的数字
    public int findRepeatNumber(final int[] nums) {
        final int[] count = new int[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            if (count[nums[i]] == 1) {
                return nums[i];
            } else {
                ++count[nums[i]];
            }
        }
        return 0;
    }

    // 剑指 Offer 03. 数组中重复的数字
    public int findRepeatNumber2(final int[] nums) {
        Arrays.sort(nums);
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] == nums[i - 1]) {
                return nums[i];
            }
        }
        return 0;
    }

    // 剑指 Offer 03. 数组中重复的数字
    public int findRepeatNumber3(final int[] nums) {
        for (int i = 0; i < nums.length; ++i) {
            nums[i] += 1;
        }
        for (int i = 0; i < nums.length; ++i) {
            if (nums[Math.abs(nums[i]) - 1] > 0) {
                nums[Math.abs(nums[i]) - 1] *= -1;
            } else {
                return Math.abs(nums[i]) - 1;
            }
        }
        return -1;
    }

    // 剑指 Offer 03. 数组中重复的数字 与 442、448、4 一块看
    public int findRepeatNumber4(final int[] nums) {
        int res = 0;
        for (int i = 0; i < nums.length; ++i) {
            int index = Math.abs(nums[i]);
            if (nums[index] >= 0) {
                nums[index] *= -1;
            } else {
                res = index;
                break;
            }
        }
        return res;

    }

    // 1128. 等价多米诺骨牌对的数量
    public int numEquivDominoPairs(final int[][] dominoes) {
        int[] dp = new int[100];
        for (int[] domino : dominoes) {
            if (domino[0] > domino[1]) {
                int temp = domino[0];
                domino[0] = domino[1];
                domino[1] = temp;
            }
            ++dp[domino[0] * 10 + domino[1]];
        }
        int result = 0;
        for (int x : dp) {
            result += x * (x - 1) / 2;
        }
        return result;

    }

    // 剑指 Offer 04. 二维数组中的查找
    public boolean findNumberIn2DArray(final int[][] matrix, final int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int i = 0;
        int j = matrix[0].length - 1;
        while (i < matrix.length && j >= 0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                --j;
            } else if (matrix[i][j] < target) {
                ++i;
            }
        }
        return false;

    }

    // 剑指 Offer 53 - I. 在排序数组中查找数字 I
    public int search53_I(final int[] nums, final int target) {
        int leftIndex = findLeftIndex(nums, target);
        int rightIndex = findRightIndex(nums, target);
        if (leftIndex == -1 || rightIndex == -1) {
            return 0;
        }
        return rightIndex - leftIndex + 1;

    }

    private int findRightIndex(int[] nums, int target) {
        int left = 0;
        int ans = -1;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] == target) {
                ans = mid;
                left = mid + 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            }
        }
        return ans;
    }

    private int findLeftIndex(int[] nums, int target) {
        int ans = -1;
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] == target) {
                ans = mid;
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else if (nums[mid] > target) {
                right = mid - 1;
            }
        }
        return ans;
    }

    // 1365. 有多少小于当前数字的数字
    public int[] smallerNumbersThanCurrent(final int[] nums) {
        int[] counts = new int[101];
        for (int num : nums) {
            ++counts[num];
        }
        for (int i = 1; i < counts.length; ++i) {
            counts[i] += counts[i - 1];
        }
        int[] res = new int[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            res[i] = nums[i] == 0 ? 0 : counts[nums[i] - 1];
        }
        return res;

    }

    // 1351. 统计有序矩阵中的负数
    public int countNegatives(final int[][] grid) {
        int count = 0;
        int i = 0;
        int j = grid[0].length - 1;
        while (i < grid.length && j >= 0) {
            if (grid[i][j] < 0) {
                count += grid.length - i;
                --j;
            } else {
                ++i;
            }
        }
        return count;

    }

    // 1337. 矩阵中战斗力最弱的 K 行
    public int[] kWeakestRows(final int[][] mat, final int k) {
        boolean[] flag = new boolean[mat.length];
        int[] result = new int[k];
        int index = 0;
        for (int i = 0; i < mat[0].length; ++i) {
            for (int j = 0; j < mat.length; ++j) {
                if (mat[j][i] == 0 && !flag[j]) {
                    flag[j] = true;
                    result[index++] = j;
                }
                if (index == k) {
                    return result;
                }
            }
        }
        for (int i = 0; i < flag.length; ++i) {
            if (!flag[i]) {
                result[index++] = i;
            }
            if (index == k) {
                return result;
            }
        }
        return result;

    }

    // 1338. 数组大小减半
    public int minSetSize(final int[] arr) {
        int[] counts = new int[100001];
        for (int num : arr) {
            ++counts[num];
        }
        Arrays.sort(counts);
        int count = 0;
        for (int i = counts.length - 1; i >= 0; --i) {
            count += counts[i];
            if (count * 2 >= arr.length) {
                return counts.length - i;

            }

        }
        return 0;

    }

    // 面试题 08.03. 魔术索引
    public int findMagicIndex(final int[] nums) {
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == i) {
                return i;
            }
        }
        return -1;

    }

    // 1333. 餐厅过滤器
    public List<Integer> filterRestaurants(int[][] restaurants, int veganFriendly, int maxPrice, int maxDistance) {
        for (int[] restaurant : restaurants) {
            if ((veganFriendly == 1 && restaurant[2] == 0) || restaurant[3] > maxPrice || restaurant[4] > maxDistance) {
                restaurant[1] = 0;
            }
        }
        Arrays.sort(restaurants, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[1] == o2[1]) {
                    return o2[0] - o1[0];
                }
                return o2[1] - o1[1];
            }
        });
        List<Integer> res = new ArrayList<>();
        for (int[] restaurant : restaurants) {
            if (restaurant[1] != 0) {
                res.add(restaurant[0]);
            }
        }
        return res;

    }

    // 1346. 检查整数及其两倍数是否存在
    public boolean checkIfExist(final int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : arr) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (entry.getKey() == 0) {
                if (entry.getValue() >= 2) {
                    return true;
                }
            } else if (map.containsKey(entry.getKey() * 2)) {
                return true;
            }
        }
        return false;
    }

    // 1346. 检查整数及其两倍数是否存在
    public boolean checkIfExist2(final int[] arr) {
        int[] neg = new int[2001];
        int[] pos = new int[2001];
        for (int i = 0; i < arr.length; ++i) {
            if (arr[i] < 0) {
                ++neg[-arr[i]];
            } else {
                ++pos[arr[i]];
            }
        }
        for (int i = 0; i < 1001; ++i) {
            if (neg[i] > 0 && neg[i * 2] > 0) {
                return true;
            }
        }
        for (int i = 0; i < 1001; ++i) {
            if (i == 0 && pos[i] > 1) {
                return true;
            } else if (i != 0 && pos[i] > 0 && pos[i * 2] > 0) {
                return true;
            }
        }
        return false;

    }

    // 1352. 最后 K 个数的乘积
    class ProductOfNumbers {
        List<Integer> list;

        public ProductOfNumbers() {
            list = new ArrayList<>();
            list.add(1);
        }

        public void add(final int num) {
            if (num == 0) {
                list = new ArrayList<>();
                list.add(1);
            } else {
                list.add(list.get(list.size() - 1) * num);
            }
        }

        public int getProduct(final int k) {
            if (k >= list.size()) {
                return 0;
            }
            return list.get(list.size() - 1) / list.get(list.size() - 1 - k);
        }
    }

    // 1366. 通过投票对团队排名
    public String rankTeams(final String[] votes) {
        final int[][] dp = new int[27][27];
        final StringBuilder builder = new StringBuilder();
        for (final String vote : votes) {
            for (int i = 0; i < vote.length(); ++i) {
                ++dp[vote.charAt(i) - 'A'][i];
                dp[vote.charAt(i) - 'A'][26] = 26 - (vote.charAt(i) - 'A');
            }
        }
        Arrays.sort(dp, new Comparator<int[]>() {
            @Override
            public int compare(final int[] o1, final int[] o2) {
                for (int i = 0; i < o1.length; ++i) {
                    if (o1[i] != o2[i]) {
                        return o2[i] - o1[i];
                    }
                }
                return 0;
            }
        });
        for (int i = 0; i < dp.length; ++i) {
            if (dp[i][26] != 0) {
                builder.append((char) (26 - (dp[i][26] - 'A')));
            }
        }
        return builder.toString();
    }

    // 面试题 16.06. 最小差
    public int smallestDifference(final int[] a, final int[] b) {
        Arrays.sort(a);
        Arrays.sort(b);
        int i = 0;
        int j = 0;
        long min = Integer.MAX_VALUE;
        while (i < a.length && j < b.length) {
            if (a[i] == b[j]) {
                return 0;
            }
            min = Math.min(min, Math.abs((long) a[i] - (long) b[j]));
            if (a[i] < b[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        return (int) min;

    }

    public int search_10_03(final int[] arr, final int target) {
        final int rotateIndex = findRotateIndex(arr, 0, arr.length - 1);
        if (rotateIndex == 0) {
            return binarySearch(arr, 0, arr.length - 1, target);
        }
        if (arr[rotateIndex] == target) {
            return rotateIndex;
        }
        if (target < arr[0]) {
            return binarySearch(arr, rotateIndex, arr.length - 1, target);
        } else {
            return binarySearch(arr, 0, rotateIndex - 1, target);
        }

    }

    private int binarySearch(final int[] arr, int left, int right, final int target) {
        while (left <= right) {
            final int mid = left + ((right - left) >>> 1);
            if (arr[mid] == target) {
                return mid;
            } else if (arr[mid] > target) {
                right = mid - 1;
            } else if (arr[mid] < target) {
                left = mid + 1;
            }
        }
        return -1;
    }

    private int findRotateIndex(final int[] arr, int left, int right) {
        if (arr[left] < arr[right]) {
            return 0;
        }
        final int mid = left + ((right - left) >>> 1);
        while (left <= right) {
            if (arr[mid] > arr[mid + 1]) {
                return mid + 1;
            }
            if (arr[mid] < arr[left]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return 0;
    }

    // 面试题 16.10. 生存人数
    public int maxAliveYear(final int[] birth, final int[] death) {
        final int[] counts = new int[200];
        for (int i = 0; i < birth.length; ++i) {
            for (int live = birth[i] - 1900; live <= death[i] - 1900; ++live) {
                ++counts[live];
            }
        }
        int maxYear = -1;
        int maxCount = counts[0];
        for (int i = 0; i < counts.length; ++i) {
            if (counts[i] > maxCount) {
                maxCount = counts[i];
                maxYear = i;
            }
        }
        return maxYear + 1900;

    }

    // 面试题 16.10. 生存人数
    public int maxAliveYear2(int[] birth, int[] death) {
        int[] diff = new int[120];
        for (int i = 0; i < birth.length; ++i) {
            ++diff[birth[i] - 1900];
            --diff[death[i] - 1900 + 1];
        }
        int maxPeople = diff[0];
        int maxIndex = 0;
        for (int i = 1; i < diff.length; ++i) {
            diff[i] += diff[i - 1];
            if (diff[i] > maxPeople) {
                maxPeople = diff[i];
                maxIndex = i;
            }
        }
        return maxIndex + 1900;

    }

    // 面试题 16.16. 部分排序
    public int[] subSort(final int[] array) {
        if (array == null || array.length == 0) {
            return new int[] { -1, -1 };
        }
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int i = 1; i < array.length; ++i) {
            if (array[i - 1] > array[i]) {
                min = Math.min(min, array[i]);
            }
        }
        for (int i = array.length - 2; i >= 0; --i) {
            if (array[i] > array[i + 1]) {
                max = Math.max(max, array[i]);
            }
        }
        int left = array.length - 1;
        int right = 0;
        for (int i = 0; i < array.length; ++i) {
            if (array[i] > min) {
                left = i;
                break;
            }
        }
        for (int i = array.length - 1; i >= 0; --i) {
            if (array[i] < max) {
                right = i;
                break;
            }
        }
        return right > left ? new int[] { left, right } : new int[] { -1, -1 };

    }

    // 面试题 16.15. 珠玑妙算
    public int[] masterMind(final String solution, final String guess) {
        int[] ans = new int[2];
        int[] counts = new int[4];
        for (int i = 0; i < solution.length(); ++i) {
            if (solution.charAt(i) == guess.charAt(i)) {
                ++ans[0];
            }
            if (solution.charAt(i) == 'R') {
                ++counts[0];
            } else if (solution.charAt(i) == 'G') {
                ++counts[1];
            } else if (solution.charAt(i) == 'B') {
                ++counts[2];
            } else if (solution.charAt(i) == 'Y') {
                ++counts[3];
            }
        }
        for (int i = 0; i < guess.length(); ++i) {
            if (guess.charAt(i) == 'R' && counts[0] > 0) {
                --counts[0];
                ++ans[1];
            } else if (guess.charAt(i) == 'G' && counts[1] > 0) {
                --counts[1];
                ++ans[1];
            } else if (guess.charAt(i) == 'B' && counts[2] > 0) {
                --counts[2];
                ++ans[1];
            } else if (guess.charAt(i) == 'Y' && counts[3] > 0) {
                --counts[3];
                ++ans[1];
            }
        }
        ans[1] -= ans[0];
        return ans;

    }

    // 1144. 递减元素使数组呈锯齿状
    public int movesToMakeZigzag(final int[] nums) {
        int res = 0;
        if (nums == null || nums.length < 3) {
            return res;
        }
        // 判断偶数位 将偶数位上的数减小到比两侧的奇数位的数都小
        int count = 0;
        int temp;
        for (int i = 0; i < nums.length; i += 2) {
            temp = nums[i];
            // 若当前偶数位的数 >= 它左边奇数位的数 ，那么将当前的数减小到它左边的数 - 1 ，注意使用temp存储该减小以后的值，目的是不改变数组
            if (i > 0 && temp >= nums[i - 1]) {
                count += temp - nums[i - 1] + 1;
                temp = nums[i - 1] - 1;
            }
            // 若当前偶数位的数（该数已经存储在temp中，可能已经被上面的if更新） >= 它右边奇数位的数 ，那么将当前的数减小到它右边的数 - 1
            if (i < nums.length - 1 && temp >= nums[i + 1]) {
                count += temp - nums[i + 1] + 1;
                // nums[i] = nums[i + 1] - 1;
            }
        }
        res = count;
        count = 0;
        // 判断奇数位 将奇数位上的数减小到比两侧的偶数位的数都小
        for (int i = 1; i < nums.length; i += 2) {
            // 若当前奇数位的数 >= 它左边偶数位的数 ，那么将当前的数减小到它左边的数 - 1
            if (i > 0 && nums[i] >= nums[i - 1]) {
                count += nums[i] - nums[i - 1] + 1;
                nums[i] = nums[i - 1] - 1;
            }
            if (i < nums.length - 1 && nums[i] >= nums[i + 1]) {
                count += nums[i] - nums[i + 1] + 1;
                // nums[i] = nums[i + 1] - 1;
            }
        }
        res = Math.min(res, count);
        return res;
    }

    class SnapshotArray {
        private final List<TreeMap<Integer, Integer>> mData;
        private int snap = 0;

        public SnapshotArray(final int length) {
            mData = new ArrayList<>();
            for (int i = 0; i < length; ++i) {
                mData.add(new TreeMap<>());
            }
        }

        public void set(final int index, final int val) {
            final TreeMap<Integer, Integer> treeMap = mData.get(index);
            treeMap.put(snap, val);
        }

        public int snap() {
            return snap++;
        }

        public int get(final int index, final int snap_id) {
            final TreeMap<Integer, Integer> treeMap = mData.get(index);
            final Integer key = treeMap.floorKey(snap_id);
            return key == null ? 0 : treeMap.get(key);
        }
    }

    // 169. 多数元素
    public int majorityElement7(final int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];

    }

    // 169. 多数元素
    public int majorityElement6(final int[] nums) {
        int count = 0;
        int majority = nums[0];
        for (int i = 0; i < nums.length; ++i) {
            if (count == 0) {
                majority = nums[i];
            }
            count += nums[i] == majority ? 1 : -1;
        }
        return majority;

    }

    // 1013. 将数组分成和相等的三个部分
    public boolean canThreePartsEqualSum(final int[] A) {
        int sum = 0;
        for (int num : A) {
            sum += num;
        }
        int count = 0;
        int curSum = 0;
        int i = 0;
        while (i < A.length && count < 3) {
            curSum += A[i];
            if (curSum * 3 == sum) {
                curSum = 0;
                ++count;
            }
            ++i;
        }
        return count == 3;

    }

    // 1170. 比较字符串最小字母出现频次
    public int[] numSmallerByFrequency(final String[] queries, final String[] words) {
        int[] counts = new int[12];
        for (int i = 0; i < words.length; ++i) {
            ++counts[checkNumSmaller(words[i])];
        }
        for (int i = counts.length - 2; i >= 0; --i) {
            counts[i] += counts[i + 1];
        }
        int[] res = new int[queries.length];
        for (int i = 0; i < res.length; ++i) {
            res[i] = counts[checkNumSmaller(queries[i]) + 1];
        }
        return res;

    }

    private int checkNumSmaller(String string) {
        char min = 'z';
        int count = 0;
        for (char c : string.toCharArray()) {
            if (c < min) {
                min = c;
                count = 1;
            } else if (c == min) {
                ++count;
            }
        }
        return count;
    }

    // 1331. 数组序号转换
    public int[] arrayRankTransform(final int[] arr) {
        int[] arrCopy = arr.clone();
        Arrays.sort(arrCopy);
        int index = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : arrCopy) {
            map.put(num, map.get(num) == null ? ++index : index);
        }
        for (int i = 0; i < arr.length; ++i) {
            arr[i] = map.get(arr[i]);
        }
        return arr;

    }

    // 1184. 公交站间的距离
    public int distanceBetweenBusStops2(final int[] distance, int start, int destination) {
        if (start > destination) {
            int temp = start;
            start = destination;
            destination = temp;
        }
        int result = 0;
        for (int i = start; i < destination; ++i) {
            result += distance[i];
        }
        return Math.min(result, Arrays.stream(distance).sum() - result);

    }

    public String dayOfTheWeek(final int day, int month, int year) {
        final Map<Integer, String> map = new HashMap<>();
        map.put(0, "Monday");
        map.put(1, "Tuesday");
        map.put(2, "Wednesday");
        map.put(3, "Thursday");
        map.put(4, "Friday");
        map.put(5, "Saturday");
        map.put(6, "Sunday");
        // 吉姆拉尔森公式
        if (month == 1 || month == 2) {
            month += 12;
            year--;
        }
        final int iWeek = (day + 2 * month + 3 * (month + 1) / 5 + year + year / 4 - year / 100 + year / 400) % 7;
        return map.get(iWeek);

    }

    public List<List<Integer>> minimumAbsDifference(final int[] arr) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(arr);
        int min = Integer.MAX_VALUE;
        for (int i = 1; i < arr.length; ++i) {
            min = Math.min(min, arr[i] - arr[i - 1]);
        }
        for (int i = 1; i < arr.length; ++i) {
            if (arr[i] - arr[i - 1] == min) {
                result.add(Arrays.asList(arr[i - 1], arr[i]));
            }
        }

        return result;
    }

    public int minCostToMoveChips(final int[] chips) {
        int odd = 0;
        int even = 0;
        for (int i = 0; i < chips.length; ++i) {
            if (chips[i] % 2 == 0) {
                ++even;
            } else {
                ++odd;
            }
        }
        return Math.min(even, odd);

    }

    // 1222. 可以攻击国王的皇后
    public List<List<Integer>> queensAttacktheKing(final int[][] queens, final int[] king) {
        final int N = 8;
        final List<List<Integer>> result = new ArrayList<>();
        final Set<Integer> set = new HashSet<>();
        for (int i = 0; i < queens.length; ++i) {
            set.add(queens[i][0] * 10 + queens[i][1]);
        }
        // 从king的位置往↖遍历
        int r = king[0];
        int c = king[1];
        while (--r >= 0 && --c >= 0) {
            if (set.contains(r * 10 + c)) {
                result.add(Arrays.asList(r, c));
                break;
            }
        }
        // 从king的位置往↘遍历
        r = king[0];
        c = king[1];
        while (++r < N && ++c < N) {
            if (set.contains(r * 10 + c)) {
                result.add(Arrays.asList(r, c));
                break;
            }
        }
        // 从king的位置往↙遍历
        r = king[0];
        c = king[1];
        while (++r < N && --c >= 0) {
            if (set.contains(r * 10 + c)) {
                result.add(Arrays.asList(r, c));
                break;
            }
        }
        // 从king的位置往↗遍历
        r = king[0];
        c = king[1];
        while (--r >= 0 && ++c < N) {
            if (set.contains(r * 10 + c)) {
                result.add(Arrays.asList(r, c));
                break;
            }
        }
        // 从king的位置往↑遍历
        r = king[0];
        c = king[1];
        while (--r >= 0) {
            if (set.contains(r * 10 + c)) {
                result.add(Arrays.asList(r, c));
                break;
            }
        }
        // 从king的位置往↓遍历
        r = king[0];
        c = king[1];
        while (++r < N) {
            if (set.contains(r * 10 + c)) {
                result.add(Arrays.asList(r, c));
                break;
            }
        }

        // 从king的位置往左遍历
        r = king[0];
        c = king[1];
        while (--c >= 0) {
            if (set.contains(r * 10 + c)) {
                result.add(Arrays.asList(r, c));
                break;
            }
        }
        // 从king的位置往→遍历
        r = king[0];
        c = king[1];
        while (++c < N) {
            if (set.contains(r * 10 + c)) {
                result.add(Arrays.asList(r, c));
                break;
            }
        }
        return result;

    }

    public boolean checkStraightLine(final int[][] coordinates) {

        int starX = coordinates[0][0];
        int starY = coordinates[0][1];
        for (int i = 0; i < coordinates.length; ++i) {
            coordinates[i][0] -= starX;
            coordinates[i][1] -= starY;
        }

        int A = -coordinates[1][1];
        int B = coordinates[1][0];
        for (int i = 2; i < coordinates.length; ++i) {
            if (A * coordinates[i][0] + B * coordinates[i][1] != 0) {
                return false;
            }
        }
        return true;

    }

    // 超时
    public int oddCells(final int n, final int m, final int[][] indices) {
        int result = 0;
        if (n == 0 || m == 0) {
            return result;
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (isOddCell(i, j, indices)) {
                    ++result;
                }
            }
        }
        return result;

    }

    // 输入：当前坐标位置
    // 输出：总和是否为奇数
    private boolean isOddCell(int i, final int j, final int[][] indices) {
        int count = 0;
        for (int k = 0; k < indices.length; ++k) {
            if (indices[k][0] == i) {
                ++count;
            }
            if (indices[k][1] == j) {
                ++count;
            }
        }
        return count % 2 == 1;
    }

    // 未超时 暴力法
    public int oddCells2(final int n, final int m, final int[][] indices) {
        int count = 0;
        if (n == 0 || m == 0) {
            return count;
        }
        final int[][] array = new int[n][m];
        for (int i = 0; i < indices.length; ++i) {
            final int r = indices[i][0];
            final int c = indices[i][1];
            // r行自增1
            for (int j = 0; j < m; ++j) {
                ++array[r][j];
            }
            // c列自增1
            for (int k = 0; k < n; ++k) {
                ++array[k][c];
            }
        }
        for (int i = 0; i < array.length; ++i) {
            for (int j = 0; j < array[0].length; ++j) {
                if (array[i][j] % 2 == 1) {
                    ++count;
                }
            }
        }
        return count;
    }

    // 辅助数组 分别统计行和列
    public int oddCells3(final int n, final int m, final int[][] indices) {
        int count = 0;
        if (n == 0 || m == 0) {
            return count;
        }
        final int[] r = new int[n];
        final int[] c = new int[m];
        for (int i = 0; i < indices.length; ++i) {
            ++r[indices[i][0]];
            ++c[indices[i][1]];
        }
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if ((r[i] + c[j]) % 2 == 1) {
                    ++count;
                }
            }
        }
        return count;

    }

    // 利用 奇数 + 偶数 = 奇数 的特性
    public int oddCells4(final int n, final int m, final int[][] indices) {
        int[] rCount = new int[m];
        int[] cCount = new int[n];
        for (int[] indice : indices) {
            ++rCount[indice[0]];
            ++cCount[indice[1]];
        }
        int oddR = 0;
        int oddC = 0;
        for (int count : rCount) {
            if (count % 2 == 1) {
                ++oddR;
            }
        }
        for (int count : cCount) {
            if (count % 2 == 1) {
                ++oddC;
            }
        }
        return oddR * (n - oddC) + oddC * (m - oddR);
    }

    // 1260. 二维网格迁移
    public List<List<Integer>> shiftGrid3(final int[][] grid, final int k) {
        int[][] result = new int[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                result[(i + (j + k) / grid[0].length) % grid.length][(j + k) % grid[0].length] = grid[i][j];
            }
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < result.length; ++i) {
            List<Integer> list = new ArrayList<>();
            for (int j = 0; j < result[0].length; ++j) {
                list.add(result[i][j]);
            }
            res.add(list);
        }
        return res;

    }

    // 1266. 访问所有点的最小时间
    public int minTimeToVisitAllPoints(final int[][] points) {
        int res = 0;
        for (int i = 1; i < points.length; ++i) {
            res += Math.max(Math.abs(points[i][1] - points[i - 1][1]), Math.abs(points[i][0] - points[i - 1][0]));
        }
        return res;

    }

    // 1275. 找出井字棋的获胜者
    public String tictactoe(final int[][] moves) {
        int[] counts = new int[8];
        for (int i = moves.length - 1; i >= 0; i -= 2) {
            ++counts[moves[i][0]];
            ++counts[moves[i][1] + 3];
            if (moves[i][0] == moves[i][1]) {
                ++counts[6];
            }
            if (moves[i][0] + moves[i][1] == 2) {
                ++counts[7];
            }

        }
        for (int count : counts) {
            if (count == 3) {
                return moves.length % 2 == 1 ? "A" : "B";
            }
        }
        if (moves.length == 9) {
            return "Draw";
        }
        return "Pending";

    }

    // 面试题 16.04. 井字游戏
    public String tictactoe(String[] board) {
        int[][] counts = new int[board.length * 2 + 2][2];
        boolean hasSlot = false;
        for (int i = 0; i < board.length; ++i) {
            for (int j = 0; j < board[0].length(); ++j) {
                if (board[i].charAt(j) == 'O') {
                    ++counts[i][0];
                    ++counts[j + board.length][0];
                    if (i == j) {
                        ++counts[counts.length - 2][0];
                    }
                    if (i + j == board.length - 1) {
                        ++counts[counts.length - 1][0];
                    }
                } else if (board[i].charAt(j) == 'X') {
                    ++counts[i][1];
                    ++counts[j + board.length][1];
                    if (i == j) {
                        ++counts[counts.length - 2][1];
                    }
                    if (i + j == board.length - 1) {
                        ++counts[counts.length - 1][1];
                    }
                } else {
                    hasSlot = true;
                }
            }
        }
        for (int i = 0; i < counts.length; ++i) {
            for (int j = 0; j < counts[i].length; ++j) {
                if (counts[i][j] == board.length) {
                    if (j == 0) {
                        return "O";
                    } else if (j == 1) {
                        return "X";
                    }
                }
            }
        }
        if (hasSlot) {
            return "Pending";
        }
        return "Draw";

    }

    // 1304. 和为零的N个唯一整数
    public int[] sumZero(final int n) {
        final int[] result = new int[n];
        for (int i = 1; i < n; ++i) {
            result[i] = i;
        }
        result[0] = ((1 - n) * n) >>> 1;
        return result;
    }

    // 1304. 和为零的N个唯一整数
    public int[] sumZero2(final int n) {
        int[] result = new int[n];
        int sum = 0;
        for (int i = 1; i < n; ++i) {
            result[i] = i;
            sum += i;
        }
        result[0] = -sum;
        return result;
    }

    // 1304. 和为零的N个唯一整数
    public int[] sumZero3(final int n) {
        int[] result = new int[n];
        int index = 0;
        for (int i = -(n / 2); i <= (n / 2); ++i) {
            if (i == 0) {
                continue;
            }
            result[index++] = i;
        }
        // if (n % 2 == 1) {
        // result[result.length - 1] = 0;
        // }

        return result;
    }

    // 1752. 检查数组是否经排序和轮转得到
    public boolean check(int[] nums) {
        int count = 0;
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i - 1] > nums[i]) {
                ++count;
            }
        }
        return (count == 0) || (count == 1 && nums[0] >= nums[nums.length - 1]);

    }

    // 1672. 最富有客户的资产总量
    public int maximumWealth(int[][] accounts) {
        int max = 0;
        for (int[] account : accounts) {
            max = Math.max(max, Arrays.stream(account).sum());
        }
        return max;

    }

    // 1652. 拆炸弹
    public int[] decrypt(int[] code, int k) {
        int[] res = new int[code.length];
        if (k == 0) {
            return res;
        } else if (k > 0) {
            for (int i = 0; i < code.length; ++i) {
                int sum = 0;
                for (int j = 1; j <= k; ++j) {
                    sum += code[(i + j) % code.length];
                }
                res[i] = sum;
            }
        } else {
            for (int i = 0; i < code.length; ++i) {
                int sum = 0;
                for (int j = -1; j >= k; --j) {
                    sum += code[((i + j) % code.length + code.length) % code.length];
                }
                res[i] = sum;
            }
        }
        return res;

    }

    // 1646. 获取生成数组中的最大值
    public int getMaximumGenerated(int n) {
        if (n == 0) {
            return 0;
        }
        int[] res = new int[n + 1];
        res[0] = 0;
        res[1] = 1;
        int max = 1;
        for (int i = 1; i < (n + 1) / 2; ++i) {
            res[i * 2] = res[i];
            res[i * 2 + 1] = res[i] + res[i + 1];
            max = Math.max(max, Math.max(res[i * 2], res[i * 2 + 1]));
        }
        return max;
    }

    // 78. 子集 / 面试题 08.04. 幂集

    public List<List<Integer>> subsets(final int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        for (int i = (1 << nums.length); i < (1 << (nums.length + 1)); ++i) {
            String bitMask = Integer.toBinaryString(i).substring(1);
            List<Integer> list = new ArrayList<>();
            for (int j = 0; j < nums.length; ++j) {
                if (bitMask.charAt(j) == '1') {
                    list.add(nums[j]);
                }
            }
            res.add(list);
        }
        return res;

    }

    // 1329. 将矩阵按对角线排序
    public int[][] diagonalSort(final int[][] mat) {
        Map<Integer, ArrayList<Integer>> map = new HashMap<>();
        for (int i = 0; i < mat.length; ++i) {
            for (int j = 0; j < mat[0].length; ++j) {
                ArrayList<Integer> list = null;
                if (map.get(i - j) == null) {
                    list = new ArrayList<>();
                } else {
                    list = map.get(i - j);
                }
                list.add(mat[i][j]);
                map.put(i - j, list);
            }
        }
        for (ArrayList<Integer> list : map.values()) {
            Collections.sort(list);
        }
        for (int i = 0; i < mat.length; ++i) {
            for (int j = 0; j < mat[0].length; ++j) {
                mat[i][j] = map.get(i - j).remove(0);
            }
        }
        return mat;

    }

    // 1380. 矩阵中的幸运数
    public List<Integer> luckyNumbers(final int[][] matrix) {
        int[] rowMin = new int[matrix.length];
        Arrays.fill(rowMin, Integer.MAX_VALUE);
        int[] colMax = new int[matrix[0].length];
        Arrays.fill(colMax, Integer.MIN_VALUE);

        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[0].length; ++j) {
                rowMin[i] = Math.min(rowMin[i], matrix[i][j]);
                colMax[j] = Math.max(colMax[j], matrix[i][j]);
            }
        }
        Set<Integer> set = new HashSet<>();
        for (int num : rowMin) {
            set.add(num);
        }
        List<Integer> list = new ArrayList<>();
        for (int num : colMax) {
            if (set.contains(num)) {
                list.add(num);
            }
        }
        return list;

    }

    public String compressString2(final String S) {
        if (S.isEmpty()) {
            return S;
        }
        int count = 1;
        final StringBuilder builder = new StringBuilder();
        builder.append(S.charAt(0));
        for (int i = 1; i < S.length(); ++i) {
            if (S.charAt(i) == builder.charAt(builder.length() - 1)) {
                ++count;
            } else {
                builder.append(count);
                builder.append(S.charAt(i));
                count = 1;
            }
        }
        builder.append(count);
        return builder.length() < S.length() ? builder.toString() : S;

    }

    public int sumNums(final int n) {
        return (1 + n) * n / 2;
    }

    public int sumNums2(final int n) {
        int result = 0;
        final boolean b = n > 0 && ((result = n + sumNums2(n - 1)) > 0);
        return result;
    }

    public int singleNumber(final int[] nums) {
        final Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        for (final int a : map.keySet()) {
            if (map.get(a) == 1) {
                return a;
            }
        }
        return 0;
    }

    public int singleNumber2(final int[] nums) {
        Arrays.sort(nums);
        if (nums[0] != nums[1]) {
            return nums[0];
        }
        for (int i = 1; i < nums.length - 1; ++i) {
            if (nums[i] != nums[i - 1] && nums[i] != nums[i + 1]) {
                return nums[i];
            }
        }
        return nums[nums.length - 1];

    }

    // 1399. 统计最大组的数目
    public int countLargestGroup(int n) {
        int[] counts = new int[37];
        for (int i = 1; i <= n; ++i) {
            ++counts[getSumByBit(i)];
        }
        int result = 0;
        int max = Arrays.stream(counts).max().getAsInt();
        for (int i = 0; i < counts.length; ++i) {
            if (counts[i] == max) {
                ++result;
            }
        }
        return result;

    }

    private int getSumByBit(int num) {
        int sum = 0;
        while (num != 0) {
            int mod = num % 10;
            num /= 10;
            sum += mod;
        }
        return sum;
    }

    // 1413. 逐步求和得到正数的最小值
    public int minStartValue(int[] nums) {
        for (int i = 1; i < nums.length; ++i) {
            nums[i] += nums[i - 1];
        }
        int min = Arrays.stream(nums).min().getAsInt();
        return min > 0 ? 1 : 1 - min;

    }

    // 1431. 拥有最多糖果的孩子
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int max = Arrays.stream(candies).max().getAsInt();
        List<Boolean> res = new ArrayList<>();
        for (int candie : candies) {
            res.add(candie + extraCandies >= max);
        }
        return res;

    }

    // 1437. 是否所有 1 都至少相隔 k 个元素
    public boolean kLengthApart(int[] nums, int k) {
        int pre = -1;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == 1) {
                if (pre != -1 && i - pre - 1 < k) {
                    return false;
                }
                pre = i;
            }
        }
        return true;

    }

    // 1450. 在既定时间做作业的学生人数
    public int busyStudent(int[] startTime, int[] endTime, int queryTime) {
        int count = 0;
        for (int i = 0; i < startTime.length; ++i) {
            if (startTime[i] <= queryTime && queryTime <= endTime[i]) {
                ++count;
            }
        }
        return count;

    }

    // 1460. 通过翻转子数组使两个数组相等
    public boolean canBeEqual(int[] target, int[] arr) {
        Arrays.sort(target);
        Arrays.sort(arr);
        for (int i = 0; i < arr.length; ++i) {
            if (target[i] != arr[i]) {
                return false;
            }
        }
        return true;

    }

    // 1460. 通过翻转子数组使两个数组相等
    public boolean canBeEqual2(int[] target, int[] arr) {
        int[] counts = new int[1001];
        for (int num : arr) {
            ++counts[num];
        }
        for (int num : target) {
            --counts[num];
        }
        for (int num : counts) {
            if (num != 0) {
                return false;
            }
        }
        return true;

    }

    // 1700. 无法吃午餐的学生数量
    public int countStudents(int[] students, int[] sandwiches) {
        int[] counts = new int[2];
        for (int student : students) {
            ++counts[student];
        }
        for (int sandwich : sandwiches) {
            if (counts[sandwich] == 0) {
                break;
            }
            --counts[sandwich];
        }
        return counts[0] + counts[1];

    }

    // 1640. 能否连接形成数组
    public boolean canFormArray(int[] arr, int[][] pieces) {
        int[] map = new int[101];
        Arrays.fill(map, -1);
        for (int i = 0; i < pieces.length; ++i) {
            map[pieces[i][0]] = i;
        }
        int i = 0;
        while (i < arr.length) {
            int pieceIndex = map[arr[i]];
            if (pieceIndex == -1) {
                return false;
            }
            ++i;
            int[] piece = pieces[pieceIndex];
            for (int j = 1; j < piece.length; ++j) {
                if (piece[j] != arr[i]) {
                    return false;
                }
                ++i;
            }
        }
        return true;

    }

    // 算法正确 数据类型用int不行 得用long
    public double averageWaitingTime(int[][] customers) {
        int waitTime = 0;
        for (int i = 0; i < customers.length; ++i) {
            if (i == 0) {
                customers[i][1] += customers[i][0];
            } else {
                if (customers[i - 1][1] > customers[i][0]) {
                    customers[i][1] += customers[i - 1][1];
                } else {
                    customers[i][1] += customers[i][0];
                }
            }
            waitTime += customers[i][1] - customers[i][0];

        }
        return (double) waitTime / (double) customers.length;

    }

    // 1701. 平均等待时间
    public double averageWaitingTime2(int[][] customers) {
        long[] endTime = new long[customers.length];
        for (int i = 0; i < customers.length; ++i) {
            if (i > 0 && endTime[i - 1] > customers[i][0]) {
                endTime[i] = endTime[i - 1] + customers[i][1];
            } else {
                endTime[i] = customers[i][0] + customers[i][1];
            }

        }
        for (int i = 0; i < customers.length; ++i) {
            endTime[i] -= customers[i][0];
        }
        return Arrays.stream(endTime).sum() / (double) customers.length;

    }

    // 1701. 平均等待时间
    public double averageWaitingTime3(int[][] customers) {
        // 表示每位顾客取完餐的时间
        long endTime = 0;
        long waitTime = 0;
        for (int i = 0; i < customers.length; ++i) {
            // 上一位顾客取完餐的时刻 比当前顾客来的时刻还晚 说明当前顾客来的时候厨师在忙 则：当前顾客取完餐的时刻 = 上一位顾客离开时的更晚的时刻 +
            // 当前顾客等待的时间
            if (endTime > customers[i][0]) {
                endTime += customers[i][1];
            }
            // 上一位顾客取完餐的时刻 比当前顾客来的时刻早 说明当前顾客来的时候厨师空闲 则：当前顾客取完餐的时刻 = 当前顾客来的时刻 + 当前顾客等待的时间
            else {
                endTime = customers[i][0] + customers[i][1];
            }
            waitTime += endTime - customers[i][0];
        }
        return waitTime / (double) customers.length;

    }

    // 1726. 同积元组
    public int tupleSameProduct(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            for (int j = i + 1; j < nums.length; ++j) {
                map.put(nums[i] * nums[j], map.getOrDefault(nums[i] * nums[j], 0) + 1);
            }
        }
        int count = 0;
        for (int value : map.values()) {
            count += value * (value - 1) / 2;
        }
        return count * 8;

    }

    // 1758. 生成交替二进制字符串的最少操作数
    public int minOperations1758(String s) {
        char[] chars = s.toCharArray();
        int count1 = 0;
        int count2 = 0;

        // 0101010 形式
        for (int i = 0; i < chars.length; ++i) {
            if (i % 2 == 0 && chars[i] != '0') {
                ++count1;
            }
            if (i % 2 == 1 && chars[i] != '1') {
                ++count1;
            }
        }

        // 1010101 形式
        for (int i = 0; i < chars.length; ++i) {
            if (i % 2 == 1 && chars[i] != '0') {
                ++count2;
            }
            if (i % 2 == 0 && chars[i] != '1') {
                ++count2;
            }
        }
        return Math.min(count1, count2);

    }

    // 1758. 生成交替二进制字符串的最少操作数
    public int minOperations1758_2(String s) {
        int count = 0;
        for (int i = 0; i < s.length(); ++i) {
            if ((i % 2) != (s.charAt(i) - '0')) {
                ++count;
            }
        }
        return Math.min(count, s.length() - count);

    }

    // 1491. 去掉最低工资和最高工资后的工资平均值
    public double average(int[] salary) {
        return ((double) (Arrays.stream(salary).sum() - Arrays.stream(salary).max().getAsInt()
                - Arrays.stream(salary).min().getAsInt())) / (salary.length - 2);

    }

    // 1502. 判断能否形成等差数列
    public boolean canMakeArithmeticProgression(int[] arr) {
        Arrays.sort(arr);
        for (int i = 1; i < arr.length - 1; ++i) {
            if (arr[i] - arr[i - 1] != arr[i + 1] - arr[i]) {
                return false;
            }
        }
        return true;

    }

    // 1512. 好数对的数目
    public int numIdenticalPairs(int[] nums) {
        int[] counts = new int[101];
        for (int num : nums) {
            ++counts[num];
        }
        int res = 0;
        for (int count : counts) {
            res += count * (count - 1) / 2;
        }
        return res;
    }

    // 1534. 统计好三元组
    public int countGoodTriplets(int[] arr, int a, int b, int c) {
        int count = 0;
        for (int i = 0; i < arr.length; ++i) {
            for (int j = i + 1; j < arr.length; ++j) {
                for (int k = j + 1; k < arr.length; ++k) {
                    if (Math.abs(arr[i] - arr[j]) <= a && Math.abs(arr[j] - arr[k]) <= b
                            && Math.abs(arr[i] - arr[k]) <= c) {
                        ++count;
                    }
                }
            }
        }
        return count;

    }

    public int findKthPositive(int[] arr, int k) {
        int pre = 0;
        int index = 0;
        while (index < arr.length) {
            if (arr[index] - pre > 1) {
                if (arr[index] - pre - 1 < k) {
                    k -= arr[index] - pre - 1;
                } else {
                    return pre + k;
                }
            }
            pre = arr[index++];
        }
        return arr[arr.length - 1] + k;

    }

    // 1582. 二进制矩阵中的特殊位置
    public int numSpecial(int[][] mat) {
        int[] row = new int[mat.length];
        int[] col = new int[mat[0].length];
        for (int i = 0; i < mat.length; ++i) {
            for (int j = 0; j < mat[0].length; ++j) {
                row[i] += mat[i][j];
                col[j] += mat[i][j];
            }
        }
        int count = 0;
        for (int i = 0; i < mat.length; ++i) {
            for (int j = 0; j < mat[0].length; ++j) {
                if (mat[i][j] == 1 && row[i] == 1 && col[j] == 1) {
                    ++count;
                }
            }
        }
        return count;

    }

    // 1572. 矩阵对角线元素的和
    public int diagonalSum(int[][] mat) {
        int sum = 0;
        for (int i = 0; i < mat.length; ++i) {
            for (int j = 0; j < mat[0].length; ++j) {
                if (i == j || i + j == mat.length - 1) {
                    sum += mat[i][j];
                }
            }
        }
        return sum;

    }

    // 1572. 矩阵对角线元素的和
    public int diagonalSum2(int[][] mat) {
        int sum = 0;
        int i = 0;
        int j = mat[0].length - 1;
        for (int[] row : mat) {
            sum += row[i++] + row[j--];
        }
        if (mat.length % 2 == 1) {
            sum -= mat[mat.length / 2][mat[0].length / 2];
        }
        return sum;

    }

    // 1608. 特殊数组的特征值--o(n)
    public int specialArray(int[] nums) {
        int[] counts = new int[1001];
        for (int num : nums) {
            ++counts[num];
        }
        for (int i = counts.length - 2; i >= 0; --i) {
            counts[i] += counts[i + 1];
            if (counts[i] == i) {
                return i;
            }
        }
        return -1;

    }

    // 1608. 特殊数组的特征值--o(log(n))
    public int specialArray2(int[] nums) {
        int left = 0;
        int right = 100;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            int count = 0;
            for (int num : nums) {
                if (num >= mid) {
                    ++count;
                }
            }
            if (mid == count) {
                return mid;
            } else if (mid > count) {
                right = mid - 1;
            } else if (mid < count) {
                left = mid + 1;
            }
        }
        return -1;

    }

    // 1423. 可获得的最大点数
    public int maxScore(int[] cardPoints, int k) {
        int window = cardPoints.length - k;
        int cur = 0;
        int min = 0;
        int sum = 0;
        for (int i = 0; i < window; ++i) {
            cur += cardPoints[i];
            sum += cardPoints[i];
        }
        min = cur;
        for (int i = window; i < cardPoints.length; ++i) {
            cur += cardPoints[i] - cardPoints[i - window];
            sum += cardPoints[i];
            min = Math.min(min, cur);
        }
        return sum - min;

    }

    // 1414. 和为 K 的最少斐波那契数字数目
    public int findMinFibonacciNumbers(int k) {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(1);
        while (k >= list.get(list.size() - 1)) {
            list.add(list.get(list.size() - 1) + list.get(list.size() - 2));
        }
        int count = 0;
        for (int i = list.size() - 1; i >= 0; --i) {
            if (k >= list.get(i)) {
                k -= list.get(i);
                ++count;
                if (k == 0) {
                    return count;
                }
            }
        }
        return count;

    }

    // 162. 寻找峰值
    public int findPeakElement(int[] nums) {
        for (int i = 0; i < nums.length - 1; ++i) {
            if (nums[i] > nums[i + 1]) {
                return i;
            }
        }
        return nums.length - 1;

    }

    // 162. 寻找峰值--二分查找
    public int findPeakElement2(int[] nums) {

        int left = 0;
        int right = nums.length - 1;

        while (left < right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] > nums[mid + 1]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;

    }

    public int[] constructRectangle(int area) {
        int sqrt = (int) Math.sqrt(area);
        if (sqrt * sqrt == area) {
            return new int[] { sqrt, sqrt };
        }
        for (int i = sqrt; i >= 1; --i) {
            if (area % i == 0) {
                return new int[] { area / i, i };
            }
        }
        return null;

    }

    // 1619. 删除某些元素后的数组均值
    public double trimMean(int[] arr) {
        int sum = 0;
        int count = 0;
        Arrays.sort(arr);
        for (int i = (int) (arr.length * 0.05); i < (int) (arr.length * 0.95); ++i) {
            sum += arr[i];
            ++count;
        }
        return (double) sum / count;

    }

    // 1629. 按键持续时间最长的键
    public char slowestKey(int[] releaseTimes, String keysPressed) {
        char res = keysPressed.charAt(0);
        int maxTime = releaseTimes[0];
        for (int i = 1; i < keysPressed.length(); ++i) {
            if (releaseTimes[i] - releaseTimes[i - 1] > maxTime) {
                res = keysPressed.charAt(i);
                maxTime = releaseTimes[i] - releaseTimes[i - 1];
            } else if (releaseTimes[i] - releaseTimes[i - 1] == maxTime) {
                if (keysPressed.charAt(i) > res) {
                    res = keysPressed.charAt(i);
                }
            }
        }
        return res;

    }

    public List<List<Integer>> groupThePeople(final int[] groupSizes) {
        final Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < groupSizes.length; ++i) {
            if (map.get(groupSizes[i]) == null) {
                final List<Integer> list = new ArrayList<>();
                list.add(i);
                map.put(groupSizes[i], list);
            } else {
                final List<Integer> list = map.get(groupSizes[i]);
                list.add(i);
                map.put(groupSizes[i], list);
            }
        }
        final List<List<Integer>> results = new ArrayList<>();
        for (final int size : map.keySet()) {
            final List<Integer> list = map.get(size);
            int i = 0;
            while (i < list.size()) {
                final List<Integer> result = list.subList(i, i + size);
                results.add(result);
                i += size;
            }
        }
        return results;

    }

    // 1773. 统计匹配检索规则的物品数量
    public int countMatches(List<List<String>> items, String ruleKey, String ruleValue) {
        int count = 0;
        for (List<String> item : items) {
            if ("type".equals(ruleKey) && item.get(0).equals(ruleValue)) {
                ++count;
            } else if ("color".equals(ruleKey) && item.get(1).equals(ruleValue)) {
                ++count;
            } else if ("name".equals(ruleKey) && item.get(2).equals(ruleValue)) {
                ++count;
            }
        }
        return count;

    }

    // 1773. 统计匹配检索规则的物品数量
    public int countMatches2(List<List<String>> items, String ruleKey, String ruleValue) {
        int typeIndex = -1;
        int count = 0;
        if ("type".equals(ruleKey)) {
            typeIndex = 0;
        } else if ("color".equals(ruleKey)) {
            typeIndex = 1;
        } else if ("name".equals(ruleKey)) {
            typeIndex = 2;
        }
        for (List<String> item : items) {
            if (item.get(typeIndex).equals(ruleValue)) {
                ++count;
            }
        }
        return count;

    }

    // 1769. 移动所有球到每个盒子所需的最小操作数
    public int[] minOperations(String boxes) {
        int[] left = new int[boxes.length()];
        int[] right = new int[boxes.length()];

        int count = boxes.charAt(0) - '0';
        for (int i = 1; i < boxes.length(); ++i) {
            left[i] = left[i - 1] + count;
            count += boxes.charAt(i) - '0';
        }
        count = boxes.charAt(boxes.length() - 1) - '0';
        for (int i = boxes.length() - 2; i >= 0; --i) {
            right[i] = right[i + 1] + count;
            count += boxes.charAt(i) - '0';
        }
        int[] answer = new int[boxes.length()];
        for (int i = 0; i < answer.length; ++i) {
            answer[i] = left[i] + right[i];
        }
        return answer;

    }

    // 1742. 盒子中小球的最大数量
    public int countBalls(int lowLimit, int highLimit) {
        int[] counts = new int[46];
        for (int num = lowLimit; num <= highLimit; ++num) {
            ++counts[countSumByBit(num)];
        }
        return Arrays.stream(counts).max().getAsInt();

    }

    private int countSumByBit(int num) {
        int sum = 0;
        while (num != 0) {
            sum += num % 10;
            num /= 10;
        }
        return sum;
    }

    // 31. 下一个排列
    public void nextPermutation(final int[] nums) {
        int i = nums.length - 1;
        while (i > 0) {
            if (nums[i - 1] < nums[i]) {
                break;
            }
            --i;
        }
        if (i == 0) {
            Arrays.sort(nums);
            return;
        }
        --i;
        int j = nums.length - 1;
        while (i < j) {
            if (nums[j] > nums[i]) {
                int temp = nums[j];
                nums[j] = nums[i];
                nums[i] = temp;
                break;
            }
            --j;
        }
        Arrays.sort(nums, i + 1, nums.length);

    }

    public List<String> getValidT9Words(final String num, final String[] words) {
        final Map<Character, Set<Character>> map = new HashMap<>();
        final Set<Character> set2 = new HashSet<>();
        set2.add('a');
        set2.add('b');
        set2.add('c');
        map.put('2', set2);

        final Set<Character> set3 = new HashSet<>();
        set3.add('d');
        set3.add('e');
        set3.add('f');
        map.put('3', set3);
        final Set<Character> set4 = new HashSet<>();
        set4.add('g');
        set4.add('h');
        set4.add('i');
        map.put('4', set4);
        final Set<Character> set5 = new HashSet<>();
        set5.add('j');
        set5.add('k');
        set5.add('l');
        map.put('5', set5);
        final Set<Character> set6 = new HashSet<>();
        set6.add('m');
        set6.add('n');
        set6.add('o');
        map.put('6', set6);
        final Set<Character> set7 = new HashSet<>();
        set7.add('p');
        set7.add('q');
        set7.add('r');
        set7.add('s');
        map.put('7', set7);
        final Set<Character> set8 = new HashSet<>();
        set8.add('t');
        set8.add('u');
        set8.add('v');
        map.put('8', set8);
        final Set<Character> set9 = new HashSet<>();
        set9.add('w');
        set9.add('x');
        set9.add('y');
        set9.add('z');
        map.put('9', set9);

        final boolean[] flag = new boolean[words.length];
        Arrays.fill(flag, true);

        for (int i = 0; i < num.length(); ++i) {
            final Set<Character> set = map.get(num.charAt(i));
            for (int j = 0; j < words.length; ++j) {
                if (!set.contains(words[j].charAt(i))) {
                    flag[j] = false;
                }
            }
        }
        final List<String> result = new ArrayList<>();
        for (int i = 0; i < flag.length; ++i) {
            if (flag[i]) {
                result.add(words[i]);
            }
        }
        return result;

    }

    // 1277. 统计全为 1 的正方形子矩阵
    public int countSquares(final int[][] matrix) {
        int sum = 0;
        int[][] dp = new int[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[0].length; ++j) {
                if (i == 0 || j == 0) {
                    dp[i][j] = matrix[i][j];
                } else if (matrix[i][j] == 0) {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                }
                sum += dp[i][j];
            }
        }
        return sum;

    }

    public List<Integer> pancakeSort(final int[] A) {
        List<Integer> result = new ArrayList<>();
        int temp = 0;
        for (int i = 0; i < A.length; ++i) {
            int max = A[0];
            int maxIndex = 0;
            for (int j = 0; j < A.length - temp; ++j) {
                if (A[j] > max) {
                    maxIndex = j;
                    max = A[j];
                }
            }
            if (maxIndex != A.length - temp - 1) {
                if (maxIndex != 0) {
                    result.add(maxIndex + 1);
                    reverse969(A, 0, maxIndex);
                }
                result.add(A.length - temp);
                reverse969(A, 0, A.length - temp - 1);
            }
            ++temp;

        }
        return result;

    }

    private void reverse969(int[] nums, int left, int right) {
        while (left < right) {
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            ++left;
            --right;
        }
    }

    // 1267. 统计参与通信的服务器
    public int countServers(final int[][] grid) {
        int result = 0;
        int[] rCounts = new int[grid.length];
        int[] cCounts = new int[grid[0].length];
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                rCounts[i] += grid[i][j];
                cCounts[j] += grid[i][j];
            }
        }
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                if (rCounts[i] > 1 || cCounts[j] > 1) {
                    result += grid[i][j];
                }
            }
        }

        return result;

    }

    // 面试题 17.19. 消失的两个数字
    public int[] missingTwo(final int[] nums) {
        int actualSum = 0;
        for (final int num : nums) {
            actualSum += num;
        }
        final int shouldSum = (1 + nums.length + 2) * (nums.length + 2) / 2;
        final int diff = shouldSum - actualSum;

        final int threshold = diff / 2;
        int sum = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] <= threshold) {
                sum += nums[i];
            }
        }
        final int result1 = (1 + threshold) * threshold / 2 - sum;
        return new int[] { result1, diff - result1 };

    }

    public int longestPalindrome(final String s) {
        final int[] counts = new int[60];
        for (final char c : s.toCharArray()) {
            ++counts[c - 'A'];
        }
        int count = 0;
        for (int i = 0; i < counts.length; ++i) {
            count += counts[i] - (counts[i] & 1);
        }
        return count < s.length() ? count + 1 : count;
    }

    public int[] getLeastNumbers(final int[] arr, final int k) {
        if (arr == null || arr.length == 0 || k == 0) {
            return new int[] {};
        }
        Arrays.sort(arr);
        final int[] result = new int[k];
        for (int i = 0; i < k; ++i) {
            result[i] = arr[i];
        }
        return result;
    }

    // 1389. 按既定顺序创建目标数组
    public int[] createTargetArray(final int[] nums, final int[] index) {
        List<Integer> list = new LinkedList<>();
        for (int i = 0; i < index.length; ++i) {
            list.add(index[i], nums[i]);
        }
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); ++i) {
            res[i] = list.get(i);
        }
        return res;
    }

    // 1385. 两个数组间的距离值-模拟
    public int findTheDistanceValue(final int[] arr1, final int[] arr2, final int d) {
        int count = 0;
        for (int i = 0; i < arr1.length; ++i) {
            int left = arr1[i] - d;
            int right = arr1[i] + d;
            boolean flag = false;
            for (int j = 0; j < arr2.length; ++j) {
                if (left <= arr2[j] && arr2[j] <= right) {
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                ++count;
            }
        }
        return count;

    }

    // 1385. 两个数组间的距离值--二分
    public int findTheDistanceValue2(final int[] arr1, final int[] arr2, final int d) {
        Arrays.sort(arr2);
        int count = 0;
        for (int i = 0; i < arr1.length; ++i) {
            if (getMinDistance(arr2, arr1[i]) > d) {
                ++count;
            }
        }
        return count;

    }

    private int getMinDistance(int[] nums, int target) {
        if (target <= nums[0]) {
            return Math.abs(target - nums[0]);
        } else if (target >= nums[nums.length - 1]) {
            return Math.abs(target - nums[nums.length - 1]);
        }
        int left = 0;
        int right = nums.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] == target) {
                return 0;
            } else if (nums[mid] > target) {
                right = mid - 1;
            } else if (nums[mid] < target) {
                left = mid + 1;
            }
        }
        return Math.min(Math.abs(target - nums[left]), Math.abs(target - nums[left - 1]));
    }

    // 835. 图像重叠
    public int largestOverlap(final int[][] A, final int[][] B) {
        int[][] dp = new int[A.length * 2 + 1][A.length * 2 + 1];
        for (int i = 0; i < A.length; ++i) {
            for (int j = 0; j < A[0].length; ++j) {
                if (A[i][j] == 1) {
                    for (int m = 0; m < B.length; ++m) {
                        for (int n = 0; n < B[0].length; ++n) {
                            if (B[m][n] == 1) {
                                ++dp[i - m + A.length][j - n + A.length];
                            }
                        }
                    }
                }
            }
        }
        int max = 0;
        for (int[] subDp : dp) {
            max = Math.max(max, Arrays.stream(subDp).max().getAsInt());
        }
        return max;

    }

    // 1343. 大小为 K 且平均值大于等于阈值的子数组数目
    public int numOfSubarrays(final int[] arr, final int k, final int threshold) {
        int sumThreshold = threshold * k;
        int count = 0;
        int cur = 0;
        for (int i = 0; i < k; ++i) {
            cur += arr[i];
        }
        if (cur >= sumThreshold) {
            ++count;
        }
        for (int i = k; i < arr.length; ++i) {
            cur += arr[i] - arr[i - k];
            if (cur >= sumThreshold) {
                ++count;
            }
        }
        return count;

    }

    // 1375. 灯泡开关 III
    public int numTimesAllBlue(final int[] light) {
        int count = 0;
        int max = 0;
        for (int i = 0; i < light.length; ++i) {
            max = Math.max(light[i], max);
            if (max <= i + 1) {
                ++count;
            }
        }
        return count;

    }

    public int lengthOfLongestSubstring(final String s) {
        int count = 0;
        final Set<Character> set = new HashSet<>();
        int i = 0;
        int j = 0;
        while (i < s.length() && j < s.length()) {
            if (!set.contains(s.charAt(j))) {
                set.add(s.charAt(j++));
                count = Math.max(count, j - i);
            } else {
                set.remove(s.charAt(i++));
            }
        }
        return count;

    }

    public int lengthOfLongestSubstring2(final String s) {
        int count = 0;
        final Set<Character> set = new HashSet<>();
        int i = 0;
        int j = 0;
        while (i < s.length() && j < s.length()) {
            if (!set.contains(s.charAt(j))) {
                set.add(s.charAt(j++));
                count = Math.max(count, j - i);
            } else {
                set.remove(s.charAt(i++));
            }
        }
        return count;

    }

    // 中心扩展算法
    public String longestPalindrome2(final String s) {
        if (s == null || s.length() < 1) {
            return "";
        }
        int left = 0;
        int right = 0;
        for (int i = 0; i < s.length(); ++i) {
            final int len1 = calculateConsecutive(s, i, i);
            final int len2 = calculateConsecutive(s, i, i + 1);
            final int len = Math.max(len1, len2);
            if (len > right - left) {
                left = i - (len - 1) / 2;
                right = i + len / 2;
            }
        }
        return s.substring(left, right + 1);

    }

    private int calculateConsecutive(final String s, final int left, final int right) {
        int l = left;
        int r = right;
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
            --l;
            ++r;
        }
        return r - l - 1;
    }

    public String convert(final String s, final int numRows) {
        if (numRows == 0) {
            return "";
        }
        if (s == null || numRows == 1 || s.length() <= numRows) {
            return s;
        }
        final StringBuilder builder = new StringBuilder();
        // 第0行
        for (int i = 0; i < s.length(); i += (numRows - 1) * 2) {
            builder.append(s.charAt(i));
        }
        // 第1行 ~ 第n-1行
        for (int i = 1; i < numRows - 1; ++i) {
            builder.append(s.charAt(i));
            int j = i + (numRows - 1) * 2 - i * 2;
            while (j < s.length()) {
                builder.append(s.charAt(j));
                j += i * 2;
                if (j < s.length()) {
                    builder.append(s.charAt(j));
                } else {
                    break;
                }
                j += (numRows - 1) * 2 - i * 2;
            }
        }
        // 第n行
        for (int i = numRows - 1; i < s.length(); i += (numRows - 1) * 2) {
            builder.append(s.charAt(i));
        }
        return builder.toString();

    }

    public String convert2(final String s, final int numRows) {
        if (numRows == 1) {
            return s;
        }
        final StringBuilder builder = new StringBuilder();
        final int cycleLen = (numRows - 1) * 2;
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j + i < s.length(); j += cycleLen) {
                builder.append(s.charAt(j + i));
                if (i != 0 && i != numRows - 1 && j + cycleLen - i < s.length()) {
                    builder.append(s.charAt(j + cycleLen - i));
                }
            }
        }
        return builder.toString();

    }

    public int myAtoi(final String str) {
        int index = 0;
        while (index < str.length()) {
            if (str.charAt(index) != ' ') {
                break;
            }
            ++index;
        }
        if (index == str.length()) {
            return 0;
        }

        int sign = 1;
        final char firstChar = str.charAt(index);
        if (firstChar == '+') {
            ++index;
            sign = 1;
        } else if (firstChar == '-') {
            ++index;
            sign = -1;
        }
        int res = 0;
        while (index < str.length()) {
            final char currentChar = str.charAt(index);
            if (currentChar > '9' || currentChar < '0') {
                break;
            }
            if (res > Integer.MAX_VALUE / 10
                    || ((res == Integer.MAX_VALUE / 10 && ((currentChar - '0') > Integer.MAX_VALUE % 10)))) {
                return Integer.MAX_VALUE;
            }
            if (res < Integer.MIN_VALUE / 10
                    || ((res == Integer.MIN_VALUE / 10) && ((currentChar - '0') > -(Integer.MIN_VALUE % 10)))) {
                return Integer.MIN_VALUE;
            }
            res = res * 10 + sign * (currentChar - '0');
            ++index;

        }
        return res;

    }

    public String intToRoman2(int num) {
        final StringBuilder builder = new StringBuilder();
        // 处理千位
        int temp = num / 1000;
        for (int i = 0; i < temp; ++i) {
            builder.append("M");
        }
        num %= 1000;

        // 处理百位
        temp = num / 100;
        if (temp == 9) {
            builder.append("CM");
        } else if (temp >= 5) {
            builder.append("D");
            for (int i = 0; i < temp - 5; ++i) {
                builder.append("C");
            }
        } else if (temp == 4) {
            builder.append("CD");
        } else {
            for (int i = 0; i < temp; ++i) {
                builder.append("C");
            }
        }
        num %= 100;

        // 处理十位
        temp = num / 10;
        if (temp == 9) {
            builder.append("XC");
        } else if (temp >= 5) {
            builder.append("L");
            for (int i = 0; i < temp - 5; ++i) {
                builder.append("X");
            }
        } else if (temp == 4) {
            builder.append("XL");
        } else {
            for (int i = 0; i < temp; ++i) {
                builder.append("X");
            }
        }
        num %= 10;

        // 处理个位
        if (num == 9) {
            builder.append("IX");
        } else if (num >= 5) {
            builder.append("V");
            for (int i = 0; i < num - 5; ++i) {
                builder.append("I");
            }
        } else if (num == 4) {
            builder.append("IV");
        } else {
            for (int i = 0; i < num; ++i) {
                builder.append("I");
            }
        }

        return builder.toString();
    }

    public String intToRoman(int num) {
        final int[] number = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
        final String[] s = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
        final StringBuilder sb = new StringBuilder();
        for (int i = 0; i < number.length; ++i) {
            while (num >= number[i]) {
                sb.append(s[i]);
                num -= number[i];
            }
            if (num == 0) {
                break;
            }
        }
        return sb.toString();

    }

    public int lengthOfLongestSubstring3(final String s) {
        final Set<Character> set = new HashSet<>();
        int i = 0;
        int j = 0;
        int count = 0;
        while (i < s.length() && j < s.length()) {
            if (!set.contains(s.charAt(j))) {
                set.add(s.charAt(j++));
                count = Math.max(count, j - i);
            } else {
                set.remove(s.charAt(i++));
            }
        }
        return count;

    }

    public String longestPalindrome3(final String s) {
        if (s == null || s.length() < 1) {
            return "";
        }
        int left = 0;
        int right = 0;
        for (int i = 0; i < s.length(); ++i) {
            final int len1 = checkPalindrome2(s, i, i);
            final int len2 = checkPalindrome2(s, i, i + 1);
            final int len = Math.max(len1, len2);
            if (len > right - left + 1) {
                left = i - (len - 1) / 2;
                right = i + len / 2;
            }
        }
        return s.substring(left, right + 1);

    }

    private int checkPalindrome2(final String s, final int left, final int right) {
        int l = left;
        int r = right;
        while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
            --l;
            ++r;
        }
        return r - l - 1;
    }

    public String convert3(final String s, final int numRows) {
        if (s == null || numRows == 1 || s.length() <= numRows) {
            return s;
        }
        final int cycleLen = (numRows - 1) * 2;
        final StringBuilder builder = new StringBuilder();
        for (int i = 0; i < numRows; ++i) {
            for (int j = 0; j + i < s.length(); j += cycleLen) {
                builder.append(s.charAt(j + i));
                if (i != 0 && i != numRows - 1 && j + cycleLen - i < s.length()) {
                    builder.append(s.charAt(j + cycleLen - i));
                }
            }
        }
        return builder.toString();

    }

    public int myAtoi2(final String str) {
        int index = 0;
        while (index < str.length()) {
            if (str.charAt(index) != ' ') {
                break;
            }
            ++index;
        }
        if (index == str.length()) {
            return 0;
        }
        int sign = 1;
        if (str.charAt(index) == '+') {
            ++index;
            sign = 1;
        } else if (str.charAt(index) == '-') {
            ++index;
            sign = -1;
        }
        int res = 0;
        while (index < str.length()) {
            final char currentChar = str.charAt(index);
            if (currentChar < '0' || currentChar > '9') {
                break;
            }
            if (res > Integer.MAX_VALUE / 10
                    || ((res == Integer.MAX_VALUE / 10 && ((currentChar - '0') > Integer.MAX_VALUE % 10)))) {
                return Integer.MAX_VALUE;
            }
            if (res < Integer.MIN_VALUE / 10
                    || (res == Integer.MIN_VALUE / 10 && ((currentChar - '0') > -(Integer.MIN_VALUE % 10)))) {
                return Integer.MIN_VALUE;
            }
            res = res * 10 + sign * (currentChar - '0');
            ++index;
        }
        return res;
    }

    public String intToRoman3(int num) {
        final StringBuilder builder = new StringBuilder();
        final int[] numbers = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
        final String[] chars = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I" };
        for (int i = 0; i < numbers.length; ++i) {
            while (num >= numbers[i]) {
                builder.append(chars[i]);
                num -= numbers[i];
            }
            if (num == 0) {
                break;
            }
        }
        return builder.toString();

    }

    public int romanToInt(final String s) {
        int res = 0;
        int preNum = getValue(s.charAt(0));
        for (int i = 1; i < s.length(); ++i) {
            final int num = getValue(s.charAt(i));
            if (preNum < num) {
                res -= preNum;
            } else {
                res += preNum;
            }
            preNum = num;
        }
        res += preNum;
        return res;

    }

    private int getValue(final char c) {
        switch (c) {
            case 'I':
                return 1;
            case 'V':
                return 5;
            case 'X':
                return 10;
            case 'L':
                return 50;
            case 'C':
                return 100;
            case 'D':
                return 500;
            case 'M':
                return 1000;
            default:
                return 0;

        }
    }

    public String longestCommonPrefix(final String[] strs) {
        if (strs == null || strs.length == 0) {
            return "";

        }
        final StringBuilder builder = new StringBuilder();
        final char[] firstString = strs[0].toCharArray();
        for (int i = 0; i < firstString.length; ++i) {
            boolean flag = false;
            for (int j = 1; j < strs.length; ++j) {
                if (i >= strs[j].length() || firstString[i] != strs[j].charAt(i)) {
                    flag = true;
                    break;
                }
            }
            if (flag) {
                break;
            }
            builder.append(firstString[i]);
        }
        return builder.toString();

    }

    // 20. 有效的括号
    public boolean isValid(final String s) {
        Map<Character, Character> map = new HashMap<>();
        map.put('}', '{');
        map.put(')', '(');
        map.put(']', '[');
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (map.containsValue(c)) {
                stack.push(c);
            } else {
                if (stack.isEmpty() || map.get(c) != stack.pop()) {
                    return false;
                }
            }
        }
        return stack.isEmpty();

    }

    public List<String> generateParenthesis(final int n) {
        final List<String> res = new ArrayList<>();
        backtrack(res, "", 0, 0, n);
        return res;

    }

    private void backtrack(final List<String> res, final String string, final int open, final int close, final int n) {
        if (string.length() == n * 2) {
            res.add(string);
            return;
        }
        if (open < n) {
            backtrack(res, string + "(", open + 1, close, n);
        }
        if (close < open) {
            backtrack(res, string + ")", open, close + 1, n);
        }
    }

    public int strStr(final String haystack, final String needle) {
        return haystack.indexOf(needle);
    }

    public int longestValidParentheses(final String s) {
        int maxLen = 0;
        for (int i = 0; i < s.length(); ++i) {
            for (int j = i + 2; j <= s.length(); j += 2) {
                if (isValid2(s.substring(i, j))) {
                    maxLen = Math.max(maxLen, j - i);
                }
            }
        }
        return maxLen;
    }

    private boolean isValid2(final String s) {
        final Stack<Character> stack = new Stack<>();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                stack.push(s.charAt(i));
            } else if (!stack.isEmpty() && stack.peek() == '(') {
                stack.pop();
            } else {
                return false;
            }
        }
        return stack.isEmpty();
    }

    public int longestValidParentheses2(final String s) {
        final int[] dp = new int[s.length()];
        int maxLen = 0;
        for (int i = 1; i < s.length(); ++i) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i - 1) == '(') {
                    dp[i] = (i - 2 >= 0 ? dp[i - 2] : 0) + 2;
                } else if (i - dp[i - 1] - 1 >= 0 && s.charAt(i - dp[i - 1] - 1) == '(') {
                    dp[i] = (i - dp[i - 1] - 2 >= 0 ? dp[i - dp[i - 1] - 2] : 0) + dp[i - 1] + 2;
                }
            }
            maxLen = Math.max(maxLen, dp[i]);
        }
        return maxLen;
    }

    public int longestValidParentheses3(final String s) {
        int maxLen = 0;
        final Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                stack.pop();
                if (stack.isEmpty()) {
                    stack.push(i);
                } else {
                    maxLen = Math.max(maxLen, i - stack.peek());
                }
            }
        }
        return maxLen;

    }

    public int longestValidParentheses4(final String s) {
        int maxLen = 0;
        int left = 0;
        int right = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                ++left;
            } else {
                ++right;
            }
            if (left == right) {
                maxLen = Math.max(maxLen, right * 2);
            } else if (right > left) {
                left = 0;
                right = 0;
            }
        }
        left = 0;
        right = 0;
        for (int i = s.length() - 1; i >= 0; --i) {
            if (s.charAt(i) == '(') {
                ++left;
            } else {
                ++right;
            }
            if (left == right) {
                maxLen = Math.max(maxLen, right * 2);
            } else if (left > right) {
                left = 0;
                right = 0;
            }
        }
        return maxLen;
    }

    public String countAndSay(final int n) {
        String s = "1";
        if (n == 1) {
            return s;
        }
        for (int i = 1; i < n; ++i) {
            final StringBuilder builder = new StringBuilder();
            int left = 0;
            int right = left + 1;
            while (right < s.length()) {
                if (s.charAt(right) == s.charAt(left)) {
                    ++right;
                    continue;
                }
                builder.append(right - left).append(s.charAt(left));
                left = right;
                right = left + 1;
            }
            builder.append(right - left).append(s.charAt(left));
            s = builder.toString();
        }
        return s;
    }

    public String multiply(final String num1, final String num2) {
        if ("0".equals(num1) || "0".equals(num2)) {
            return "0";
        }
        final char[] arr1 = num1.toCharArray();
        final char[] arr2 = num2.toCharArray();
        final int[] res = new int[num1.length() + num2.length()];
        for (int i = arr1.length - 1; i >= 0; --i) {
            final int c1 = arr1[i] - '0';
            for (int j = arr2.length - 1; j >= 0; --j) {
                final int c2 = arr2[j] - '0';
                res[i + j + 1] += c1 * c2;
            }
        }
        for (int i = res.length - 1; i >= 1; --i) {
            if (res[i] > 9) {
                res[i - 1] += res[i] / 10;
                res[i] = res[i] % 10;
            }
        }
        final StringBuilder builder = new StringBuilder();
        int i = 0;
        for (; i < res.length; ++i) {
            if (res[i] != 0) {
                break;
            }
        }
        for (; i < res.length; ++i) {
            builder.append(res[i]);
        }
        return builder.toString();
    }

    public List<List<String>> groupAnagrams(final String[] strs) {
        final String[] strsCopy = strs.clone();
        for (int i = 0; i < strsCopy.length; ++i) {
            final String s = strsCopy[i];
            final char[] arr = s.toCharArray();
            Arrays.sort(arr);
            strsCopy[i] = String.valueOf(arr);
        }
        final boolean[] flag = new boolean[strs.length];
        final List<List<String>> res = new ArrayList<>();
        for (int i = 0; i < strsCopy.length; ++i) {
            if (!flag[i]) {
                flag[i] = true;
                final List<String> list = new ArrayList<>();
                list.add(strs[i]);
                for (int j = i + 1; j < strsCopy.length; ++j) {
                    if (!flag[j] && strsCopy[j].equals(strsCopy[i])) {
                        flag[j] = true;
                        list.add(strs[j]);
                    }
                }
                res.add(list);
            }

        }
        return res;
    }

    public List<List<String>> groupAnagrams2(final String[] strs) {
        if (strs == null || strs.length == 0) {
            return new ArrayList<>();
        }
        final Map<String, List<String>> map = new HashMap<>();
        for (final String s : strs) {
            final char[] chars = s.toCharArray();
            Arrays.sort(chars);
            final String key = String.valueOf(chars);
            map.computeIfAbsent(key, k -> new ArrayList<>()).add(s);
            // if (!map.containsKey(key)) {
            // map.put(key, new ArrayList());
            // }
            // map.get(key).add(s);
        }
        return new ArrayList<>(map.values());

    }

    public List<List<String>> groupAnagrams3(final String[] strs) {
        final int[] counts = new int[26];
        final Map<String, List<String>> map = new HashMap<>();
        for (final String s : strs) {
            Arrays.fill(counts, 0);
            final char[] chars = s.toCharArray();
            for (final char c : chars) {
                ++counts[c - 'a'];
            }
            final StringBuilder builder = new StringBuilder();
            for (final int count : counts) {
                builder.append("#").append(count);
            }
            if (!map.containsKey(builder.toString())) {
                map.put(builder.toString(), new ArrayList<>());
            }
            map.get(builder.toString()).add(s);
        }
        return new ArrayList<>(map.values());
    }

    public int lengthOfLastWord(final String s) {
        if (s.isEmpty()) {
            return 0;
        }
        final String sTrim = s.trim();
        final char[] chars = sTrim.toCharArray();
        int count = 0;
        for (int i = chars.length - 1; i >= 0; --i) {
            if (chars[i] == ' ') {
                break;
            }
            ++count;
        }
        return count;
    }

    public String addBinary(final String a, final String b) {
        final char[] res = new char[Math.max(a.length(), b.length()) + 1];
        int indexRes = Math.max(a.length(), b.length());
        int indexA = a.length() - 1;
        int indexB = b.length() - 1;
        int carry = 0;
        while (indexA >= 0 || indexB >= 0) {
            int charA = 0;
            int charB = 0;
            if (indexA >= 0) {
                charA = a.charAt(indexA) - '0';
            }
            if (indexB >= 0) {
                charB = b.charAt(indexB) - '0';
            }
            final int resTemp = charA + charB + carry;
            carry = resTemp / 2;
            res[indexRes--] = (char) (resTemp % 2 + '0');
            --indexA;
            --indexB;
            if (indexRes == 0) {
                res[0] = (char) (carry + '0');
            }
        }
        final String result = String.valueOf(res);
        return result.charAt(0) == '0' ? result.substring(1) : result;
    }

    // 71. 简化路径
    public static String simplifyPath(final String path) {
        Stack<String> stack = new Stack<>();
        String[] paths = path.split("\\/");
        for (String p : paths) {
            if ("..".equals(p)) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
            } else if (!p.isEmpty() && !".".equals(p)) {
                stack.push(p);
            }
        }
        if (stack.isEmpty()) {
            return "/";
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < stack.size(); ++i) {
            res.append("/").append(stack.get(i));
        }
        return res.toString();

    }

    public int minDistance(final String word1, final String word2) {
        if (word1.isEmpty() || word2.isEmpty()) {
            return word1.length() + word2.length();
        }
        final int[][] dp = new int[word1.length() + 1][word2.length() + 1];
        for (int i = 0; i < word1.length() + 1; ++i) {
            dp[i][0] = i;
        }
        for (int i = 0; i < word2.length() + 1; ++i) {
            dp[0][i] = i;
        }
        for (int i = 1; i < word1.length() + 1; ++i) {
            for (int j = 1; j < word2.length() + 1; ++j) {
                final int left = dp[i][j - 1] + 1;
                final int up = dp[i - 1][j] + 1;
                int left_up = dp[i - 1][j - 1];
                if (word1.charAt(i - 1) != word2.charAt(j - 1)) {
                    left_up += 1;
                }
                dp[i][j] = Math.min(Math.min(left, up), left_up);
            }
        }
        return dp[word1.length()][word2.length()];
    }

    public String minWindow(final String s, final String t) {
        String res = "";
        if (s.isEmpty() || t.isEmpty() || s.length() < t.length()) {
            return res;
        }
        final int[] needs = new int[128];
        for (int i = 0; i < s.length(); ++i) {
            ++needs[s.charAt(i)];
        }
        final int[] window = new int[128];
        int count = 0;
        int left = 0;
        int right = 0;
        int minLen = s.length() + 1;
        while (right < s.length()) {
            char c = s.charAt(right);
            ++window[c];
            if (needs[c] > 0 && needs[c] >= window[c]) {
                ++count;
            }
            while (count == t.length()) {
                c = s.charAt(left);
                if (needs[c] > 0 && needs[c] >= window[c]) {
                    --count;
                }
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    res = s.substring(left, right + 1);
                }
                --window[c];
                ++left;
            }
            ++right;
        }
        return res;

    }

    public String minWindow2(final String s, final String t) {
        String res = "";
        if (s.isEmpty() || t.isEmpty() || s.length() < t.length()) {
            return res;
        }
        final int[] needs = new int[128];
        for (int i = 0; i < t.length(); ++i) {
            ++needs[t.charAt(i)];
        }

        int left = 0;
        int right = 0;
        final int[] window = new int[128];
        int count = 0;
        int minLen = s.length() + 1;
        while (right < s.length()) {
            char c = s.charAt(right);
            ++window[c];
            if (needs[c] > 0 && needs[c] >= window[c]) {
                ++count;
            }
            while (count == t.length()) {
                c = s.charAt(left);
                if (needs[c] > 0 && needs[c] >= window[c]) {
                    --count;
                }
                if (right - left + 1 < minLen) {
                    minLen = right - left + 1;
                    res = s.substring(left, right + 1);
                }
                --window[c];
                ++left;
            }
            ++right;
        }

        return res;

    }

    public int numDecodings(final String s) {
        if (s.charAt(0) == '0') {
            return 0;
        }
        int pre = 1;
        int cur = 1;
        for (int i = 1; i < s.length(); ++i) {
            final int temp = cur;
            if (s.charAt(i) == '0') {
                if (s.charAt(i - 1) == '1' || s.charAt(i - 1) == '2') {
                    cur = pre;
                } else {
                    return 0;
                }
            } else if (s.charAt(i - 1) == '1' || (s.charAt(i - 1) == '2' && s.charAt(i) >= '1' && s.charAt(i) <= '6')) {
                cur += pre;
            }
            pre = temp;
        }
        return cur;
    }

    public boolean isPalindrome(final String s) {
        if (s == null) {
            return false;
        }
        if (s.isEmpty()) {
            return true;
        }
        int left = 0;
        int right = s.length() - 1;
        while (right > left) {

            if (!Character.isLetterOrDigit(s.charAt(left))) {
                ++left;
                continue;
            }
            if (!Character.isLetterOrDigit(s.charAt(right))) {
                --right;
                continue;
            }
            if ((Character.isDigit(s.charAt(left)) && Character.isDigit(s.charAt(right))
                    && s.charAt(left) == s.charAt(right))
                    || (Character.isLetter(s.charAt(left)) && Character.isLetter(s.charAt(right))
                            && String.valueOf(s.charAt(left)).equalsIgnoreCase(String.valueOf(s.charAt(right))))) {
                ++left;
                --right;
                continue;
            }
            return false;
        }
        return true;
    }

    // 9. 回文数
    public boolean isPalindrome9(int x) {
        if (x < 0) {
            return false;
        }
        int xOrigin = x;
        int res = 0;
        while (x != 0) {
            int mod = x % 10;
            x /= 10;
            if (res > Integer.MAX_VALUE / 10) {
                return false;
            }
            res = res * 10 + mod;
        }
        return res == xOrigin;

    }

    public int compareVersion(final String version1, final String version2) {
        final String[] strs1 = version1.split("\\.");
        final String[] strs2 = version2.split("\\.");
        int p1 = 0;
        int p2 = 0;
        while (p1 < strs1.length || p2 < strs2.length) {
            final int a1 = p1 < strs1.length ? Integer.parseInt(strs1[p1]) : 0;
            final int a2 = p2 < strs2.length ? Integer.parseInt(strs2[p2]) : 0;
            if (a1 != a2) {
                return a1 > a2 ? 1 : -1;
            }
            ++p1;
            ++p2;
        }
        return 0;

    }

    public static String reverseWords(final String s) {
        final StringBuilder builder = new StringBuilder();
        if (s == null || s.trim().isEmpty()) {
            return builder.toString();
        }
        final String[] strs = s.split("\\ ");
        for (int i = strs.length - 1; i >= 0; --i) {
            if (!strs[i].isEmpty()) {
                builder.append(strs[i]).append(" ");
            }
        }
        return builder.toString().trim();

    }

    // 227. 基本计算器 II

    public int calculate(final String s) {
        Stack<Integer> stack = new Stack<>();
        char sign = '+';

        while (i < s.length()) {
            if (Character.isWhitespace(s.charAt(i))) {
                ++i;
                continue;
            }
            if (Character.isDigit(s.charAt(i))) {
                StringBuilder num = new StringBuilder();
                while (i < s.length() && Character.isDigit(s.charAt(i))) {
                    num.append(s.charAt(i++));
                }
                --i;
                if (sign == '+') {
                    stack.push(Integer.parseInt(num.toString()));
                } else if (sign == '-') {
                    stack.push(-1 * Integer.parseInt(num.toString()));
                } else {
                    stack.push(getNum(stack.pop(), Integer.parseInt(num.toString()), sign));
                }
            } else {
                sign = s.charAt(i);
            }

            ++i;

        }
        int res = 0;
        while (!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;

    }

    private Integer getNum(int a, int b, char sign) {
        if (sign == '*') {
            return a * b;
        } else {
            return a / b;
        }
    }

    public List<String> generateParenthesis2(final int n) {
        final List<String> list = new ArrayList<>();
        if (n == 0) {
            return list;
        }
        handleParenthesis(list, "", 0, 0, n);
        return list;

    }

    private void handleParenthesis(final List<String> list, final String res, final int left, final int right,
            final int n) {
        if (res.length() == n * 2) {
            list.add(res);
            return;
        }
        if (left < n) {
            handleParenthesis(list, res + "(", left + 1, right, n);
        }
        if (right < left) {
            handleParenthesis(list, res + ")", left, right + 1, n);
        }
    }

    public boolean isInterleave(final String s1, final String s2, final String s3) {
        if (s1 == null || s2 == null || s3 == null) {
            return false;
        }
        if (s1.length() + s2.length() != s3.length()) {
            return false;
        }
        final boolean[][] dp = new boolean[s1.length() + 1][s2.length() + 1];
        dp[0][0] = true;
        // 遍历第一行
        for (int i = 1; i < s2.length() + 1; ++i) {
            dp[0][i] = dp[0][i - 1] && s2.charAt(i - 1) == s3.charAt(i - 1);
        }
        // 遍历第一列
        for (int i = 1; i < s1.length() + 1; ++i) {
            dp[i][0] = dp[i - 1][0] && s1.charAt(i - 1) == s3.charAt(i - 1);
        }
        // 遍历其余项
        for (int i = 1; i < s1.length() + 1; ++i) {
            for (int j = 1; j < s2.length() + 1; ++j) {
                dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1))
                        || (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
            }
        }
        return dp[s1.length()][s2.length()];
    }

    public boolean isInterleave2(final String s1, final String s2, final String s3) {
        if (s1 == null || s2 == null || s3 == null) {
            return false;
        }
        if (s1.length() + s2.length() != s3.length()) {
            return false;
        }
        final boolean[] dp = new boolean[s2.length() + 1];
        for (int i = 0; i < s1.length() + 1; ++i) {
            for (int j = 0; j < s2.length() + 1; ++j) {
                if (i == 0 && j == 0) {
                    dp[j] = true;
                } else if (i == 0) {
                    dp[j] = dp[j - 1] && s2.charAt(j - 1) == s3.charAt(j - 1);
                } else if (j == 0) {
                    dp[j] = dp[j] && s1.charAt(i - 1) == s3.charAt(i - 1);
                } else {
                    dp[j] = (dp[j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1))
                            || (dp[j] && s1.charAt(i - 1) == s3.charAt(i + j - 1));
                }
            }
        }
        return dp[s2.length()];

    }

    public int numDistinct(final String s, final String t) {
        final int[][] dp = new int[t.length() + 1][s.length() + 1];
        Arrays.fill(dp[0], 1);
        for (int i = 1; i < t.length() + 1; ++i) {
            for (int j = 1; j < s.length() + 1; ++j) {
                if (t.charAt(i - 1) == s.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i][j - 1];
                } else {
                    dp[i][j] = dp[i][j - 1];

                }
            }
        }
        return dp[t.length()][s.length()];
    }

    public List<List<Integer>> palindromePairs(final String[] words) {
        final List<List<Integer>> list = new ArrayList<>();
        for (int i = 0; i < words.length; ++i) {
            for (int j = i + 1; j < words.length; ++j) {
                if (canMakePalindromePairs(words[i] + words[j])) {
                    list.add(Arrays.asList(i, j));
                }
                if (canMakePalindromePairs(words[j] + words[i])) {
                    list.add(Arrays.asList(j, i));
                }
            }
        }
        return list;

    }

    private boolean canMakePalindromePairs(final String string) {
        int left = 0;
        int right = string.length() - 1;
        while (left < right) {
            if (string.charAt(left) != string.charAt(right)) {
                return false;
            }
            ++left;
            --right;
        }
        return true;
    }

    public List<List<Integer>> palindromePairs2(final String[] words) {
        final List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < words.length; ++i) {
            for (int j = 0; j < words.length; ++j) {
                if (i == j) {
                    continue;
                }
                final String combine = words[i].concat(words[j]);
                final String reverse = new StringBuilder(combine).reverse().toString();
                if (combine.equals(reverse)) {
                    res.add(Arrays.asList(i, j));
                }
            }
        }
        return res;

    }

    public String reverseVowels(final String s) {
        final char[] strs = s.toCharArray();
        int left = 0;
        int right = strs.length - 1;
        while (left < right) {
            if (strs[left] != 'a' && strs[left] != 'e' && strs[left] != 'i' && strs[left] != 'o' && strs[left] != 'u'
                    && strs[left] != 'A' && strs[left] != 'E' && strs[left] != 'I' && strs[left] != 'O'
                    && strs[left] != 'U') {
                ++left;
                continue;
            }
            if (strs[right] != 'a' && strs[right] != 'e' && strs[right] != 'i' && strs[right] != 'o'
                    && strs[right] != 'u' && strs[right] != 'A' && strs[right] != 'E' && strs[right] != 'I'
                    && strs[right] != 'O' && strs[right] != 'U') {
                --right;
                continue;
            }
            final char temp = strs[left];
            strs[left] = strs[right];
            strs[right] = temp;
            ++left;
            --right;
        }
        return String.valueOf(strs);

    }

    public boolean canConstruct(final String ransomNote, final String magazine) {
        final int[] counts = new int[26];
        for (int i = 0; i < magazine.length(); ++i) {
            ++counts[magazine.charAt(i) - 'a'];
        }
        for (int i = 0; i < ransomNote.length(); ++i) {
            if (counts[ransomNote.charAt(i) - 'a'] == 0) {
                return false;
            }
            --counts[ransomNote.charAt(i) - 'a'];
        }
        return true;
    }

    public int firstUniqChar(final String s) {
        final int[] counts = new int[26];
        for (int i = 0; i < s.length(); ++i) {
            ++counts[s.charAt(i) - 'a'];
        }
        for (int i = 0; i < s.length(); ++i) {
            if (counts[s.charAt(i) - 'a'] == 1) {
                return i;
            }
        }
        return -1;
    }

    public String reverseWords2(final String s) {
        final String[] strs = s.trim().split(" ");
        final StringBuilder builder = new StringBuilder();
        for (int i = strs.length - 1; i >= 0; --i) {
            if (!strs[i].isEmpty()) {
                builder.append(strs[i]).append(" ");
            }
        }
        return builder.toString().trim();
    }

    public String addStrings(final String num1, final String num2) {
        final int[] res = new int[Math.max(num1.length(), num2.length()) + 1];
        int p1 = num1.length() - 1;
        int p2 = num2.length() - 1;
        int p3 = res.length - 1;
        while (p1 >= 0 || p2 >= 0) {
            final int a1 = p1 >= 0 ? num1.charAt(p1) - '0' : 0;
            final int a2 = p2 >= 0 ? num2.charAt(p2) - '0' : 0;
            res[p3--] = a1 + a2;
            if (p1 >= 0) {
                --p1;
            }
            if (p2 >= 0) {
                --p2;
            }
        }
        for (int i = res.length - 1; i >= 1; --i) {
            if (res[i] > 9) {
                res[i - 1] += res[i] / 10;
                res[i] = res[i] % 10;
            }
        }
        final StringBuilder builder = new StringBuilder();
        for (int i = 0; i < res.length; ++i) {
            if (i == 0) {
                if (res[i] != 0) {
                    builder.append(res[i]);
                }
            } else {
                builder.append(res[i]);
            }
        }
        return builder.toString();
    }

    public String addStrings2(final String num1, final String num2) {
        final StringBuilder builder = new StringBuilder();
        int p1 = num1.length() - 1;
        int p2 = num2.length() - 1;
        int carry = 0;
        while (p1 >= 0 || p2 >= 0) {
            final int a1 = p1 >= 0 ? num1.charAt(p1--) - '0' : 0;
            final int a2 = p2 >= 0 ? num2.charAt(p2--) - '0' : 0;
            final int temp = a1 + a2 + carry;
            carry = temp / 10;
            builder.append(temp % 10);
        }
        if (carry == 1) {
            builder.append(carry);
        }
        return builder.reverse().toString();

    }

    /**
     * "Hello, my name is John"
     * 
     */
    public int countSegments(String s) {
        if (s.trim().isEmpty()) {
            return 0;
        }
        s = s.trim();
        final String[] strs = s.split("\\s+");
        return strs.length;
    }

    public int countSegments2(final String s) {
        int count = 0;
        for (int i = 0; i < s.length(); ++i) {
            if ((i == 0 || s.charAt(i - 1) == ' ') && s.charAt(i) != ' ') {
                ++count;
            }
        }
        return count;

    }

    public int compress(final char[] chars) {
        final StringBuilder builder = new StringBuilder();
        builder.append(chars[0]);
        int count = 1;
        for (int i = 1; i < chars.length; ++i) {
            if (chars[i] == builder.charAt(builder.length() - 1)) {
                ++count;
            } else {
                if (count == 1) {
                    builder.append(chars[i]);
                } else {
                    builder.append(count).append(chars[i]);
                }
                count = 1;
            }
        }
        if (count != 1) {
            builder.append(count);
        }
        final char[] temp = builder.toString().toCharArray();
        if (temp.length <= chars.length) {
            System.arraycopy(temp, 0, chars, 0, temp.length);
        }
        return Math.min(builder.length(), chars.length);
    }

    public int compress2(final char[] chars) {
        int anchor = 0;
        int write = 0;
        for (int read = 0; read < chars.length; ++read) {
            if (read + 1 == chars.length || chars[read] != chars[read + 1]) {
                chars[write++] = chars[anchor];
                if (read > anchor) {
                    for (final char c : ("" + (read - anchor + 1)).toCharArray()) {
                        chars[write++] = c;
                    }

                }
                anchor = read + 1;
            }

        }
        return write;

    }

    public boolean repeatedSubstringPattern(final String s) {
        final String str = s + s;
        return str.substring(1, str.length() - 1).contains(s);
    }

    public static String validIPAddress(final String IP) {
        // 一定不是Ipv6，可能是Ipv4
        if (!IP.contains(":") && IP.contains(".")) {
            final String[] ipStrs = IP.split("\\.", -1);
            if (ipStrs == null || ipStrs.length != 4) {
                return "Neither";
            }
            for (final String str : ipStrs) {
                if (str == null || str.isEmpty()) {
                    return "Neither";
                }
                if ((str.length() > 1 && str.charAt(0) == '0') || str.length() > 3) {
                    return "Neither";
                }
                for (final char c : str.toCharArray()) {
                    if (!Character.isDigit(c)) {
                        return "Neither";
                    }
                }
                if (Integer.parseInt(str) > 255 || Integer.parseInt(str) < 0) {
                    return "Neither";
                }
            }
            return "IPv4";
        }
        // 一定不是Ipv4 ，可能是ipv6
        else if (!IP.contains(".") && IP.contains(":")) {
            final String[] ipStrs = IP.split("\\:", -1);
            if (ipStrs == null || ipStrs.length != 8) {
                return "Neither";
            }
            for (final String str : ipStrs) {
                if (str == null || str.isEmpty()) {
                    return "Neither";
                }
                if (str.length() > 4) {
                    return "Neither";
                }
                for (final char c : str.toCharArray()) {
                    if (!Character.isDigit(c) && (c > 'f' || c < 'a') && (c < 'A' || c > 'F')) {
                        return "Neither";
                    }
                }
            }
            return "IPv6";

        } else {
            return "Neither";
        }

    }

    public boolean detectCapitalUse(final String word) {
        return allUpperCase(word) || allLowerCase(word) || onlyFirstLetterIsUpperCase(word);

    }

    private boolean onlyFirstLetterIsUpperCase(final String word) {
        final boolean isFirstLetterUpperCase = Character.isUpperCase(word.charAt(0));
        if (!isFirstLetterUpperCase) {
            return false;
        }
        for (int i = 1; i < word.length(); ++i) {
            if (!Character.isLowerCase(word.charAt(i))) {
                return false;
            }
        }
        return true;
    }

    private boolean allLowerCase(final String word) {
        for (final char c : word.toCharArray()) {
            if (!Character.isLowerCase(c)) {
                return false;
            }
        }
        return true;
    }

    private boolean allUpperCase(final String word) {
        for (final char c : word.toCharArray()) {
            if (!Character.isUpperCase(c)) {
                return false;
            }
        }
        return true;
    }

    public boolean detectCapitalUse2(final String word) {
        if (word.length() < 2) {
            return true;
        }
        final boolean firstUpper = Character.isUpperCase(word.charAt(0));
        final boolean secondUpper = Character.isUpperCase(word.charAt(1));
        if (!firstUpper && secondUpper) {
            return false;
        }
        for (int i = 2; i < word.length(); ++i) {
            if (Character.isUpperCase(word.charAt(i)) != secondUpper) {
                return false;
            }
        }
        return true;

    }

    public int findLUSlength(final String a, final String b) {
        if (a.equals(b))
            return -1;
        return Math.max(a.length(), b.length());
    }

    public String complexNumberMultiply(final String a, final String b) {
        final String[] strA = a.split("\\+");
        final String[] strB = b.split("\\+");
        final int a1 = Integer.parseInt(strA[0]);
        final int a2 = Integer.parseInt(strA[1].substring(0, strA[1].length() - 1));

        final int b1 = Integer.parseInt(strB[0]);
        final int b2 = Integer.parseInt(strB[1].substring(0, strB[1].length() - 1));

        final int res1 = a1 * b1 - a2 * b2;
        final int res2 = a1 * b2 + a2 * b1;
        return res1 + "+" + res2 + "i";
    }

    public String reverseStr(final String s, final int k) {
        final StringBuilder builder = new StringBuilder();
        if (s.length() < k) {
            return builder.append(s).reverse().toString();
        }
        if (s.length() >= k && s.length() < 2 * k) {
            return builder.append(s.substring(0, k)).reverse().append(s.substring(k)).toString();
        }
        for (int i = 0; i < s.length(); i += 2 * k) {
            if (s.length() - i < k) {
                builder.append(new StringBuilder(s.substring(i)).reverse());
                break;
            } else if (s.length() - i >= k && s.length() - i < 2 * k) {
                builder.append(new StringBuilder(s.substring(i, i + k)).reverse()).append(s.substring(i + k));
                break;
            }
            builder.append(new StringBuilder(s.substring(i, i + k)).reverse()).append(s.substring(i + k, i + 2 * k));
        }
        return builder.toString();

    }

    public static int findMinDifference(final List<String> timePoints) {
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < timePoints.size(); ++i) {
            for (int j = i + 1; j < timePoints.size(); ++j) {
                min = Math.min(getMinDiff(timePoints.get(i), timePoints.get(j)), min);
            }
        }
        return min;

    }

    private static int getMinDiff(final String time1, final String time2) {
        final String[] time1Strs = time1.split("\\:");
        final int h1 = Integer.parseInt(time1Strs[0]);
        final int m1 = Integer.parseInt(time1Strs[1]);
        final String[] time2Strs = time2.split("\\:");
        final int h2 = Integer.parseInt(time2Strs[0]);
        final int m2 = Integer.parseInt(time2Strs[1]);
        final int d1 = Math.abs(h1 * 60 + m1 - h2 * 60 - m2);
        final int d2 = Math.abs(24 * 60 - d1);
        return Math.min(d1, d2);
    }

    public int findMinDifference2(final List<String> timePoints) {
        int min = Integer.MAX_VALUE;
        final int[] arr = new int[timePoints.size()];
        for (int i = 0; i < timePoints.size(); ++i) {
            arr[i] = Integer.parseInt(timePoints.get(i).substring(0, 2)) * 60
                    + Integer.parseInt(timePoints.get(i).substring(3));
        }
        Arrays.sort(arr);
        for (int i = 1; i < arr.length; ++i) {
            min = Math.min(min, Math.abs(arr[i] - arr[i - 1]));
        }
        min = Math.min(min, Math.abs(24 * 60 - (arr[arr.length - 1] - arr[0])));
        return min;

    }

    public boolean checkRecord(final String s) {
        int countA = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == 'A' && ++countA > 1) {
                return false;
            }
            if (s.charAt(i) == 'L' && ((i + 1 < s.length() && s.charAt(i + 1) == 'L'))
                    && ((i + 2 < s.length() && s.charAt(i + 2) == 'L'))) {
                return false;
            }
        }
        return true;
    }

    public String optimalDivision(final int[] nums) {
        if (nums.length == 1) {
            return String.valueOf(nums[0]);
        }
        if (nums.length == 2) {
            return nums[0] + "/" + nums[1];
        }
        final StringBuilder builder = new StringBuilder();
        builder.append(nums[0] + "/(" + nums[1]);
        for (int i = 2; i < nums.length; ++i) {
            builder.append("/" + nums[i]);
        }
        builder.append(")");
        return builder.toString();

    }

    public String reverseWords3(final String s) {
        final String[] strs = s.split("\\s+", -1);
        final StringBuilder builder = new StringBuilder();
        for (final String string : strs) {
            builder.append(new StringBuilder(string).reverse() + " ");
        }
        return builder.toString().trim();
    }

    public String reverseWords4(final String s) {
        final StringBuilder word = new StringBuilder();
        final StringBuilder result = new StringBuilder();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) != ' ') {
                word.append(s.charAt(i));
            } else {
                result.append(word.reverse() + " ");
                word.setLength(0);
            }
        }
        result.append(word.reverse());
        return result.toString();
    }

    // 1394. 找出数组中的幸运数
    public int findLucky(final int[] arr) {
        int[] counts = new int[501];
        for (int num : arr) {
            ++counts[num];
        }

        for (int i = counts.length - 1; i >= 1; --i) {
            if (counts[i] == i) {
                return i;
            }
        }
        return -1;

    }

    // 1395. 统计作战单位数
    public int numTeams(final int[] rating) {
        int res = 0;
        for (int i = 0; i < rating.length; ++i) {
            int leftSmaller = 0;
            int leftBigger = 0;
            int rightSmaller = 0;
            int rightBigger = 0;
            for (int j = 0; j < i; ++j) {
                if (rating[j] < rating[i]) {
                    ++leftSmaller;
                } else {
                    ++leftBigger;
                }
            }
            for (int j = i + 1; j < rating.length; ++j) {
                if (rating[j] < rating[i]) {
                    ++rightSmaller;
                } else {
                    ++rightBigger;
                }
            }
            res += leftSmaller * rightBigger + leftBigger * rightSmaller;

        }
        return res;

    }

    // 85. 最大矩形
    public int maximalRectangle(char[][] matrix) {
        int res = 0;
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return res;
        }
        int[] dp = new int[matrix[0].length];
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[0].length; ++j) {
                if (matrix[i][j] == '1') {
                    ++dp[j];
                } else {
                    dp[j] = 0;
                }
            }
            res = Math.max(res, getMaximal85(dp));
        }
        return res;

    }

    private int getMaximal85(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        stack.push(-1);
        int max = 0;
        for (int i = 0; i < heights.length; ++i) {
            while (stack.peek() != -1 && heights[stack.peek()] >= heights[i]) {
                int h = heights[stack.pop()];
                max = Math.max(max, (i - stack.peek() - 1) * h);
            }
            stack.push(i);
        }
        while (stack.peek() != -1) {
            int h = heights[stack.pop()];
            max = Math.max(max, (heights.length - stack.peek() - 1) * h);
        }
        return max;
    }

    // 670. 最大交换
    public int maximumSwap(int num) {
        int[] last = new int[10];
        char[] chars = String.valueOf(num).toCharArray();
        for (int i = 0; i < chars.length; ++i) {
            last[chars[i] - '0'] = i;
        }

        for (int i = 0; i < chars.length; ++i) {
            int cur = chars[i] - '0';
            for (int bit = 9; bit > cur; --bit) {
                if (last[bit] > i) {
                    swap(chars, last[bit], i);
                    return Integer.parseInt(new String(chars));
                }
            }
        }
        return num;
    }

    private void swap(char[] chars, int i, int j) {
        char temp = chars[i];
        chars[i] = chars[j];
        chars[j] = temp;
    }

    // 900. RLE 迭代器
    class RLEIterator {
        private int[] A;
        private int q;
        private int i;

        public RLEIterator(int[] A) {
            this.A = A;

        }

        public int next(int n) {
            while (i < A.length) {
                if (n + q > A[i]) {
                    n -= A[i] - q;
                    q = 0;
                    i += 2;
                } else {
                    q += n;
                    return A[i + 1];
                }
            }
            return -1;
        }
    }

    // 1475. 商品折扣后的最终价格
    public int[] finalPrices(int[] prices) {
        for (int i = 0; i < prices.length; ++i) {
            for (int j = i + 1; j < prices.length; ++j) {
                if (prices[j] <= prices[i]) {
                    prices[i] -= prices[j];
                    break;
                }
            }
        }
        return prices;

    }

    // 1475. 商品折扣后的最终价格
    public int[] finalPrices2(int[] prices) {
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < prices.length; ++i) {
            while (!stack.isEmpty() && prices[i] <= prices[stack.peek()]) {
                int index = stack.pop();
                prices[index] -= prices[i];
            }
            stack.push(i);
        }
        return prices;

    }

    public int[] frequencySort(int[] nums) {
        int[] counts = new int[201];
        for (int num : nums) {
            ++counts[num + 100];
        }
        for (int i = 0; i < nums.length; ++i) {
            nums[i] = 1000 * counts[nums[i] + 100] + (100 - nums[i]);
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; ++i) {
            nums[i] = 100 - nums[i] % 1000;
        }
        return nums;

    }

    class SubrectangleQueries {
        int[][] rectangle;

        public SubrectangleQueries(int[][] rectangle) {
            this.rectangle = rectangle;
        }

        public void updateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
            for (int i = row1; i <= row2; ++i) {
                for (int j = col1; j <= col2; ++j) {
                    rectangle[i][j] = newValue;
                }
            }
        }

        public int getValue(int row, int col) {
            return rectangle[row][col];
        }
    }

    class SubrectangleQueries2 {
        int[][] rectangle;
        List<List<Integer>> mRecords;

        public SubrectangleQueries2(int[][] rectangle) {
            this.rectangle = rectangle;
            mRecords = new ArrayList<>();
        }

        public void updateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
            List<Integer> record = new ArrayList<>();
            record.add(row1);
            record.add(col1);
            record.add(row2);
            record.add(col2);
            record.add(newValue);
            mRecords.add(record);
        }

        public int getValue(int row, int col) {
            for (int i = mRecords.size() - 1; i >= 0; --i) {
                List<Integer> record = mRecords.get(i);
                if (record.get(0) <= row && row <= record.get(2) && record.get(1) <= col && col <= record.get(3)) {
                    return record.get(4);
                }
            }
            return rectangle[row][col];
        }
    }

    // 1476. 子矩形查询
    class SubrectangleQueries3 {
        private int[][] mRectangle;
        private int[][] mRecords;
        private int index;

        public SubrectangleQueries3(int[][] rectangle) {
            mRectangle = rectangle;
            mRecords = new int[501][5];
        }

        public void updateSubrectangle(int row1, int col1, int row2, int col2, int newValue) {
            mRecords[index][0] = row1;
            mRecords[index][1] = col1;
            mRecords[index][2] = row2;
            mRecords[index][3] = col2;
            mRecords[index][4] = newValue;
            ++index;

        }

        public int getValue(int row, int col) {
            for (int i = index - 1; i >= 0; --i) {
                if (mRecords[i][0] <= row && row <= mRecords[i][2] && mRecords[i][1] <= col && col <= mRecords[i][3]) {
                    return mRecords[i][4];
                }
            }
            return mRectangle[row][col];
        }
    }

    // 1409. 查询带键的排列
    public int[] processQueries(int[] queries, int m) {
        List<Integer> list = new ArrayList<>();
        int[] res = new int[queries.length];
        for (int i = 0; i < m; ++i) {
            list.add(i + 1);
        }
        for (int i = 0; i < queries.length; ++i) {
            int value = queries[i];
            int index = -1;
            for (int j = 0; j < list.size(); ++j) {
                if (list.get(j) == value) {
                    index = j;
                    break;
                }
            }
            res[i] = index;
            list.remove(index);
            list.add(0, value);
        }
        return res;

    }

    public List<List<Integer>> pairSums(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int temp = nums[left] + nums[right];
            if (temp == target) {
                res.add(Arrays.asList(nums[left], nums[right]));
                ++left;
                --right;
            } else if (temp > target) {
                --right;
            } else {
                ++left;
            }
        }
        return res;

    }

    // 198. 打家劫舍
    public int rob(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; ++i) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }
        return dp[dp.length - 1];

    }

    // 213. 打家劫舍 II
    public int rob213II(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int res1 = getRob(Arrays.copyOfRange(nums, 0, nums.length - 1));
        int res2 = getRob(Arrays.copyOfRange(nums, 1, nums.length));
        return Math.max(res1, res2);
    }

    private int getRob(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; ++i) {
            dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
        }

        return dp[dp.length - 1];
    }

    // 179. 最大数
    public String largestNumber(int[] nums) {
        String[] strings = new String[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            strings[i] = String.valueOf(nums[i]);
        }
        Arrays.sort(strings, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                String order1 = o1 + o2;
                String order2 = o2 + o1;
                return order2.compareTo(order1);
            }
        });
        if ("0".equals(strings[0])) {
            return "0";
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < strings.length; ++i) {
            res.append(strings[i]);
        }

        return res.toString();

    }

    // 1689. 十-二进制数的最少数目
    public int minPartitions(String n) {
        int max = 0;
        for (char a : n.toCharArray()) {
            max = Math.max(max, a - '0');
        }
        return max;

    }

    // 322. 零钱兑换
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i < dp.length; ++i) {
            for (int coin : coins) {
                if (i >= coin) {
                    dp[i] = Math.min(dp[i], dp[i - coin] + 1);
                }
            }
        }
        return dp[dp.length - 1] > amount ? -1 : dp[dp.length - 1];

    }

    // 1656. 设计有序流
    class OrderedStream {
        private String[] stream;

        private int ptr;

        public OrderedStream(int n) {
            stream = new String[n];

        }

        public List<String> insert(int idKey, String value) {
            stream[idKey - 1] = value;
            List<String> list = new ArrayList<>();
            while (ptr < stream.length && stream[ptr] != null) {
                list.add(stream[ptr++]);
            }
            return list;

        }

    }

    // 1588. 所有奇数长度子数组的和
    public int sumOddLengthSubarrays(int[] arr) {
        int sum = 0;
        int n = arr.length;
        for (int window = 1; window <= n; window += 2) {
            int curSum = 0;
            for (int i = 0; i < window; ++i) {
                curSum += arr[i];
            }
            sum += curSum;
            for (int i = 0; i < arr.length - window; ++i) {
                curSum += arr[i + window] - arr[i];
                sum += curSum;
            }
        }
        return sum;

    }

    // 1588. 所有奇数长度子数组的和-o(n)
    public int sumOddLengthSubarrays2(int[] arr) {
        int sum = 0;
        for (int i = 0; i < arr.length; ++i) {
            int leftEvenCount = i / 2 + 1;
            int leftOddCount = (i + 1) / 2;
            int rightEvenCount = (arr.length + 1 - i) / 2;
            int rightOddCount = (arr.length - i) / 2;
            sum += arr[i] * (leftEvenCount * rightEvenCount + leftOddCount * rightOddCount);

        }
        return sum;

    }

    // 1493. 删掉一个元素以后全为 1 的最长子数组
    public int longestSubarray(int[] nums) {
        int[] left = new int[nums.length];
        int[] right = new int[nums.length];
        left[0] = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            left[i] = nums[i] == 0 ? 0 : left[i - 1] + 1;
        }
        right[nums.length - 1] = nums[nums.length - 1];
        for (int i = nums.length - 2; i >= 0; --i) {
            right[i] = nums[i] == 0 ? 0 : right[i + 1] + 1;
        }
        int max = 0;
        for (int i = 0; i < nums.length; ++i) {
            int l = i == 0 ? 0 : left[i - 1];
            int r = i == nums.length - 1 ? 0 : right[i + 1];
            max = Math.max(max, l + r);
        }
        return max;

    }

    // 1493. 删掉一个元素以后全为 1 的最长子数组

    public int longestSubarray2(int[] nums) {
        int pre0[] = new int[nums.length];
        int pre1[] = new int[nums.length];
        pre0[0] = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            pre0[i] = nums[i] == 0 ? 0 : pre0[i - 1] + 1;
        }
        int max = 0;
        pre1[0] = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            pre1[i] = nums[i] == 0 ? pre0[i - 1] : pre1[i - 1] + 1;
            max = Math.max(max, pre1[i]);
        }
        if (max == nums.length) {
            return max - 1;

        }
        return max;

    }

    // 1493. 删掉一个元素以后全为 1 的最长子数组
    public int longestSubarray3(int[] nums) {
        int pre0 = 0;
        int pre1 = 0;
        int max = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == 1) {
                ++pre0;
                ++pre1;
            } else {
                pre1 = pre0;
                pre0 = 0;
            }
            max = Math.max(max, pre1);
        }
        return max == nums.length ? max - 1 : max;

    }

    // 1481. 不同整数的最少数目
    public int findLeastNumOfUniqueInts(int[] arr, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : arr) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        List<Integer> counts = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            counts.add(entry.getValue());
        }
        Collections.sort(counts);
        for (int i = 0; i < counts.size(); ++i) {
            if (k >= counts.get(i)) {
                k -= counts.get(i);
            } else {
                return counts.size() - i;
            }
        }
        return 0;

    }

    // 1823. 找出游戏的获胜者
    public int findTheWinner(int n, int k) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            list.add(i + 1);
        }
        int i = 0;
        while (list.size() != 1) {
            i = (i + k - 1) % list.size();
            list.remove(i);
        }
        return list.get(0);

    }

    // 1438. 绝对差不超过限制的最长连续子数组
    public int longestSubarray(int[] nums, int limit) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        int count = 0;
        int res = 0;
        for (int i = 0; i < nums.length; ++i) {
            max = Math.max(max, nums[i]);
            min = Math.min(min, nums[i]);
            if (Math.abs(max - min) <= limit) {
                ++count;
            } else {
                res = Math.max(res, count);
                count = 1;
                max = nums[i];
                min = nums[i];
            }

        }
        res = Math.max(res, count);
        return res;

    }

    // 1007. 行相等的最少多米诺旋转
    public int minDominoRotations(int[] A, int[] B) {
        int rotation = checkDomino(A, B, A[0]);
        if (rotation != -1 || A[0] == B[0]) {
            return rotation;
        }
        return checkDomino(B, A, B[0]);

    }

    private int checkDomino(int[] A, int[] B, int num) {
        int rotateA = 0;
        int rotateB = 0;
        for (int i = 0; i < A.length; ++i) {
            if (A[i] != num && B[i] != num) {
                return -1;
            }
            if (A[i] != num) {
                ++rotateA;
            } else if (B[i] != num) {
                ++rotateB;
            }

        }
        return Math.min(rotateA, rotateB);
    }

    // 1552. 两球之间的磁力
    public int maxDistance(int[] position, int m) {
        Arrays.sort(position);
        int left = 1;
        int right = position[position.length - 1] - position[0];
        int ans = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (canPutAllBalls(position, mid, m)) {
                ans = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return ans;

    }

    private boolean canPutAllBalls(int[] position, int distance, int m) {
        int pre = position[0];
        int balls = 1;
        for (int i = 1; i < position.length; ++i) {
            if (position[i] - pre >= distance) {
                ++balls;
                pre = position[i];
            }
        }
        return balls >= m;
    }

    // 1560. 圆形赛道上经过次数最多的扇区
    public List<Integer> mostVisited(int n, int[] rounds) {
        int last = rounds[rounds.length - 1];
        int first = rounds[0];
        List<Integer> res = new ArrayList<>();
        if (last >= first) {
            for (int i = first; i <= last; ++i) {
                res.add(i);
            }
        } else {
            for (int i = 1; i <= last; ++i) {
                res.add(i);
            }
            for (int i = first; i <= n; ++i) {
                res.add(i);
            }
        }
        return res;

    }

    // 1442. 形成两个异或相等数组的三元组数目
    public int countTriplets(int[] arr) {
        int count = 0;
        for (int i = 0; i < arr.length; ++i) {
            int xor = 0;
            for (int k = i; k < arr.length; ++k) {
                xor ^= arr[k];
                if (xor == 0) {
                    count += k - i;
                }
            }
        }
        return count;

    }

    // 1509. 三次操作后最大值与最小值的最小差
    public int minDifference(int[] nums) {
        if (nums == null || nums.length <= 4) {
            return 0;
        }
        int res = Integer.MAX_VALUE;
        Arrays.sort(nums);
        for (int i = 0; i < 4; ++i) {
            res = Math.min(res, nums[nums.length + i - 4] - nums[i]);
        }
        return res;

    }

    public static int minDifference2(int[] nums) {
        if (nums.length <= 4) {
            return 0;
        }
        int[] min = new int[4];
        Arrays.fill(min, Integer.MAX_VALUE);
        int[] max = new int[4];
        Arrays.fill(max, Integer.MIN_VALUE);
        for (int num : nums) {
            if (num >= max[0]) {
                max[3] = max[2];
                max[2] = max[1];
                max[1] = max[0];
                max[0] = num;
            } else if (num >= max[1]) {
                max[3] = max[2];
                max[2] = max[1];
                max[1] = num;
            } else if (num >= max[2]) {
                max[3] = max[2];
                max[2] = num;
            } else if (num >= max[3]) {
                max[3] = num;
            }

            if (num <= min[0]) {
                min[3] = min[2];
                min[2] = min[1];
                min[1] = min[0];
                min[0] = num;
            } else if (num <= min[1]) {
                min[3] = min[2];
                min[2] = min[1];
                min[1] = num;
            } else if (num <= min[2]) {
                min[3] = min[2];
                min[2] = num;
            } else if (num <= min[3]) {
                min[3] = num;
            }
        }
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < max.length; ++i) {
            res = Math.min(res, max[i] - min[3 - i]);
        }
        return res;

    }

    // 1386. 安排电影院座位
    public int maxNumberOfFamilies2(int n, int[][] reservedSeats) {
        int mask1 = 0b00001111;
        int mask2 = 0b11110000;
        int mask3 = 0b11000011;
        Map<Integer, Integer> map = new HashMap<>();
        for (int[] reservedSeat : reservedSeats) {
            if (reservedSeat[1] >= 2 && reservedSeat[1] <= 9) {
                int origin = map.getOrDefault(reservedSeat[0], 0);
                int value = origin | (1 << (reservedSeat[1] - 2));
                map.put(reservedSeat[0], value);
            }
        }
        int result = (n - map.size()) * 2;
        for (int bitMask : map.values()) {
            if ((bitMask | mask1) == mask1 || (bitMask | mask2) == mask2 || (bitMask | mask3) == mask3) {
                ++result;
            }
        }
        return result;

    }

    public int[] findSwapValues(int[] array1, int[] array2) {
        int sum1 = Arrays.stream(array1).sum();
        int sum2 = Arrays.stream(array2).sum();
        int delta = sum1 - sum2;
        Set<Integer> set = new HashSet<>();
        for (int num : array1) {
            set.add(2 * num);
        }
        for (int i = 0; i < array2.length; ++i) {
            if (set.contains(2 * array2[i] + delta)) {
                return new int[] { (2 * array2[i] + delta) / 2, array2[i] };
            }
        }
        return new int[] {};

    }

    // 1524. 和为奇数的子数组数目
    public int numOfSubarrays(int[] arr) {
        final int MOD = 1000000007;
        int preSumOdd = 0;
        int preSumEven = 1;
        int res = 0;
        int sum = 0;
        for (int i = 0; i < arr.length; ++i) {
            sum += arr[i];
            if (sum % 2 == 0) {
                res = (res + preSumOdd) % MOD;
                ++preSumEven;
            } else {
                res = (res + preSumEven) % MOD;
                ++preSumOdd;
            }

        }
        return res % MOD;

    }

    // 1535. 找出数组游戏的赢家
    public int getWinner(int[] arr, int k) {
        int winner = Math.max(arr[0], arr[1]);
        int consecutiveWins = 1;
        for (int i = 2; i < arr.length; ++i) {
            if (consecutiveWins == k) {
                return winner;
            }
            if (arr[i] > winner) {
                consecutiveWins = 1;
                winner = arr[i];
            } else {
                ++consecutiveWins;
            }

        }
        return winner;

    }

    // 1292. 元素和小于等于阈值的正方形的最大边长
    public int maxSideLength(int[][] mat, int threshold) {
        int[][] P = new int[mat.length + 1][mat[0].length + 1];
        for (int i = 1; i < P.length; ++i) {
            for (int j = 1; j < P[0].length; ++j) {
                P[i][j] = P[i - 1][j] + P[i][j - 1] + mat[i - 1][j - 1] - P[i - 1][j - 1];
            }
        }
        int left = 1;
        int ans = 0;
        int right = Math.min(mat.length, mat[0].length);
        while (left <= right) {
            boolean find = false;
            int mid = left + ((right - left) >> 1);
            for (int i = 1; i <= mat.length + 1 - mid; ++i) {
                for (int j = 1; j <= mat[0].length + 1 - mid; ++j) {
                    if (getRect(P, i, j, i + mid - 1, j + mid - 1) <= threshold) {
                        find = true;
                    }

                }
            }
            if (find) {
                ans = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return ans;
    }

    // 1292. 元素和小于等于阈值的正方形的最大边长
    public int maxSideLength2(int[][] mat, int threshold) {
        int[][] P = new int[mat.length + 1][mat[0].length + 1];
        for (int i = 1; i < P.length; ++i) {
            for (int j = 1; j < P[0].length; ++j) {
                P[i][j] = P[i - 1][j] + P[i][j - 1] + mat[i - 1][j - 1] - P[i - 1][j - 1];
            }
        }
        int ans = 0;
        int r = Math.min(mat.length, mat[0].length);
        for (int i = 1; i < P.length; ++i) {
            for (int j = 1; j < P[0].length; ++j) {
                for (int side = ans + 1; side <= r; ++side) {
                    if (i + side - 1 < P.length && j + side - 1 < P[0].length
                            && getRect(P, i, j, i + side - 1, j + side - 1) <= threshold) {
                        ++ans;
                    } else {
                        break;
                    }
                }
            }
        }
        return ans;

    }

    private int getRect(int[][] P, int startI, int startJ, int endI, int endJ) {
        return P[endI][endJ] - P[endI][startJ - 1] - P[startI - 1][endJ] + P[startI - 1][startJ - 1];
    }

    // 1806. 还原排列的最少操作步数---超时
    public int reinitializePermutation(int n) {
        int ans = 0;
        int[] perm = new int[n];
        for (int i = 0; i < n; ++i) {
            perm[i] = i;
        }
        while (true) {
            ++ans;
            for (int i = 0; i < n; ++i) {
                if (i % 2 == 0) {
                    perm[i] = perm[i / 2];
                } else {
                    perm[i] = perm[n / 2 + (i - 1) / 2];
                }
            }
            boolean flag = false;
            for (int i = 0; i < n; ++i) {
                if (perm[i] != i) {
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                return ans;
            }
        }
        // return ans;

    }

    // 1738. 找出第 K 大的异或坐标值
    public int kthLargestValue(int[][] matrix, int k) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[0].length; ++j) {
                if (i == 0 && j > 0) {
                    matrix[i][j] ^= matrix[i][j - 1];
                } else if (j == 0 && i > 0) {
                    matrix[i][j] ^= matrix[i - 1][j];
                } else if (i > 0 && j > 0) {
                    matrix[i][j] ^= matrix[i - 1][j] ^ matrix[i][j - 1] ^ matrix[i - 1][j - 1];
                }
                list.add(matrix[i][j]);
            }
        }
        Collections.sort(list);
        return list.get(list.size() - k);

    }

    // 1827. 最少操作使数组递增
    public int minOperations(int[] nums) {
        int count = 0;
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i - 1] >= nums[i]) {
                count += nums[i - 1] - nums[i] + 1;
                nums[i] = nums[i - 1] + 1;
            }
        }
        return count;

    }

    // 1566. 重复至少 K 次且长度为 M 的模式
    // arr = [1,2,1,2,1,2,1,3], m = 2, k = 3
    public boolean containsPattern(int[] arr, int m, int k) {
        for (int i = 0; i <= arr.length - m * k; ++i) {
            int offset = 0;
            for (; offset < m * k; ++offset) {
                if (arr[i + offset] != arr[i + offset % m]) {
                    break;
                }
            }
            if (offset == m * k) {
                return true;
            }
        }
        return false;

    }

    // 1497. 检查数组对是否可以被 k 整除
    public boolean canArrange(int[] arr, int k) {
        int[] counts = new int[k];
        for (int num : arr) {
            ++counts[((num % k) + k) % k];
        }

        for (int i = 1; i < counts.length; ++i) {
            if (counts[i] != counts[k - i]) {
                return false;
            }
        }
        return counts[0] % 2 == 0;

    }

    // 1497. 检查数组对是否可以被 k 整除
    public boolean canArrange2(int[] arr, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : arr) {
            map.put((num % k + k) % k, map.getOrDefault((num % k + k) % k, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            int key = entry.getKey();
            int value = entry.getValue();
            if (key > 0 && value != map.getOrDefault(k - key, 0)) {
                return false;
            }
        }
        return map.getOrDefault(0, 0) % 2 == 0;

    }

    // 1465. 切割后面积最大的蛋糕
    public int maxArea(int h, int w, int[] horizontalCuts, int[] verticalCuts) {
        final int MOD = 1000000007;
        Arrays.sort(horizontalCuts);
        Arrays.sort(verticalCuts);
        long maxHeight = Math.max(h - horizontalCuts[horizontalCuts.length - 1], horizontalCuts[0]);
        long maxWidth = Math.max(w - verticalCuts[verticalCuts.length - 1], verticalCuts[0]);
        for (int i = 1; i < horizontalCuts.length; ++i) {
            maxHeight = Math.max(maxHeight, horizontalCuts[i] - horizontalCuts[i - 1]);
        }
        for (int i = 1; i < verticalCuts.length; ++i) {
            maxWidth = Math.max(maxWidth, verticalCuts[i] - verticalCuts[i - 1]);
        }
        return (int) ((maxHeight * maxWidth) % MOD);

    }

    // 1208. 尽可能使字符串相等
    public int equalSubstring(String s, String t, int maxCost) {
        int[] diff = new int[s.length()];
        for (int i = 0; i < diff.length; ++i) {
            diff[i] = Math.abs(s.charAt(i) - t.charAt(i));
        }
        int start = 0;
        int end = 0;
        int currCost = 0;
        int res = 0;
        while (end < diff.length) {
            currCost += diff[end];
            while (currCost > maxCost) {
                currCost -= diff[start];
                ++start;
            }
            res = Math.max(res, end - start + 1);
            ++end;

        }
        return res;

    }

    // 1233. 删除子文件夹
    public List<String> removeSubfolders(String[] folder) {
        Arrays.sort(folder);
        List<String> res = new ArrayList<>();
        res.add(folder[0]);
        for (int i = 1; i < folder.length; ++i) {
            String pre = res.get(res.size() - 1) + "/";
            if (folder[i].indexOf(pre) == -1) {
                res.add(folder[i]);
            }
        }
        return res;

    }

    // 1296. 划分数组为连续数字的集合
    public boolean isPossibleDivide(int[] nums, int k) {
        if (nums.length % k != 0) {
            return false;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; ++i) {
            if (map.getOrDefault(nums[i], 0) == 0) {
                continue;
            }
            for (int j = nums[i]; j < nums[i] + k; ++j) {
                if (map.getOrDefault(j, 0) == 0) {
                    return false;
                }
                map.put(j, map.get(j) - 1);
            }
        }
        return true;

    }

    // 846. 一手顺子
    public boolean isNStraightHand(int[] hand, int W) {
        if (hand.length % W != 0) {
            return false;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : hand) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        Arrays.sort(hand);
        for (int i = 0; i < hand.length; ++i) {
            if (map.getOrDefault(hand[i], 0) == 0) {
                continue;
            }
            for (int num = hand[i]; num < hand[i] + W; ++num) {
                if (map.getOrDefault(num, 0) == 0) {
                    return false;
                }
                map.put(num, map.getOrDefault(num, 0) - 1);
            }
        }
        return true;

    }

    // 1833. 雪糕的最大数量
    public int maxIceCream(int[] costs, int coins) {
        Arrays.sort(costs);
        for (int i = 0; i < costs.length; ++i) {
            if (costs[i] <= coins) {
                coins -= costs[i];
            } else {
                return i;
            }
        }
        return costs.length;

    }

    // 1833. 雪糕的最大数量
    public int maxIceCream2(int[] costs, int coins) {
        int[] counts = new int[100001];
        for (int num : costs) {
            ++counts[num];
        }
        int res = 0;
        for (int i = 1; i < counts.length; ++i) {
            if (coins >= counts[i] * i) {
                res += counts[i];
                coins -= counts[i] * i;
            } else {
                res += coins / i;
                return res;
            }
        }
        return costs.length;

    }

    // 221. 最大正方形
    public int maximalSquare(char[][] matrix) {
        int[][] dp = new int[matrix.length][matrix[0].length];
        int max = 0;
        for (int i = 0; i < matrix.length; ++i) {
            for (int j = 0; j < matrix[0].length; ++j) {
                if (i == 0 || j == 0) {
                    dp[i][j] = matrix[i][j] - '0';
                } else {
                    if (matrix[i][j] == '1') {
                        dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i][j - 1], dp[i - 1][j])) + 1;
                    }
                }
                max = Math.max(max, dp[i][j]);
            }
        }
        return max * max;
    }

    // 1482. 制作 m 束花所需的最少天数
    public int minDays(int[] bloomDay, int m, int k) {
        int ans = -1;
        int low = 1;
        int high = Arrays.stream(bloomDay).max().getAsInt();
        while (low <= high) {
            int mid = low + ((high - low) >>> 1);
            if (canMakeFlowers(bloomDay, mid, m, k)) {
                ans = mid;
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        return ans;

    }

    private boolean canMakeFlowers(int[] bloomDay, int days, int m, int k) {
        int countK = 0;
        int countM = 0;
        for (int i = 0; i < bloomDay.length; ++i) {
            if (bloomDay[i] <= days) {
                ++countK;
                if (countK == k) {
                    ++countM;
                    countK = 0;
                }
            } else {
                countK = 0;
            }
        }
        return countM >= m;

    }

    // 875. 爱吃香蕉的珂珂
    public int minEatingSpeed(int[] piles, int h) {
        int res = Integer.MAX_VALUE;
        int left = 1;
        int right = Integer.MAX_VALUE;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (canEatBananas(piles, mid, h)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private boolean canEatBananas(int[] piles, int speed, int h) {
        int spendHours = 0;
        for (int i = 0; i < piles.length; ++i) {
            spendHours += (piles[i] - 1) / speed + 1;
        }
        return spendHours <= h;
    }

    // 1424.对角线遍历 II
    public int[] findDiagonalOrder(List<List<Integer>> nums) {
        int count = 0;
        Map<Integer, List<Integer>> map = new TreeMap<>();
        for (int i = 0; i < nums.size(); ++i) {
            for (int j = 0; j < nums.get(i).size(); ++j) {
                if (!map.containsKey(i + j)) {
                    List<Integer> list = new ArrayList<>();
                    map.put(i + j, list);
                }
                map.get(i + j).add(nums.get(i).get(j));
                ++count;
            }
        }
        int[] res = new int[count];
        int index = 0;
        for (List<Integer> list : map.values()) {
            for (int i = list.size() - 1; i >= 0; --i) {
                res[index++] = list.get(i);
            }
        }
        return res;

    }

    // 面试题 17.05. 字母与数字
    public String[] findLongestSubarray(String[] array) {
        int[] dp = new int[array.length];
        for (int i = 0; i < array.length; ++i) {
            dp[i] = Character.isLetter(array[i].charAt(0)) ? 1 : -1;
            if (i > 0) {
                dp[i] += dp[i - 1];
            }
        }

        int maxStartIndex = -1;
        int max = Integer.MIN_VALUE;

        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < dp.length; ++i) {
            if (dp[i] == 0) {
                map.put(dp[i], i);
            } else {
                if (!map.containsKey(dp[i])) {
                    map.put(dp[i], i);
                } else {
                    if (max < i - map.get(dp[i])) {
                        max = i - map.get(dp[i]);
                        maxStartIndex = map.get(dp[i]);
                    }
                }
            }
        }
        if (map.get(0) == null && max == Integer.MIN_VALUE) {
            return new String[] {};
        }
        if (map.get(0) + 1 >= max) {
            return Arrays.copyOfRange(array, 0, map.get(0) + 1);
        } else {
            ++maxStartIndex;
            return Arrays.copyOfRange(array, maxStartIndex, maxStartIndex + max);
        }

    }

    // 1814. 统计一个数组中好对子的数目
    public int countNicePairs(int[] nums) {
        final int MOD = 1000000007;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            int res = num - rev(num);
            map.put(res, map.getOrDefault(res, 0) + 1);
        }
        long count = 0;
        for (long value : map.values()) {
            count = (count + value * (value - 1) / 2) % MOD;
        }
        return (int) count % MOD;

    }

    private int rev(int num) {
        int res = 0;
        while (num > 0) {
            int mod = num % 10;
            num /= 10;
            res = res * 10 + mod;
        }
        return res;

    }

    // 1711. 大餐计数
    public int countPairs(int[] deliciousness) {
        final int MOD = 1000000007;
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int delicious : deliciousness) {
            for (int j = 0; j <= 21; ++j) {
                int remain = (int) Math.pow(2, j) - delicious;
                if (remain >= 0) {
                    res = (res + map.getOrDefault(remain, 0)) % MOD;
                }
            }
            map.put(delicious, map.getOrDefault(delicious, 0) + 1);
        }
        return res % MOD;

    }

    // 1764. 通过连接另一个数组的子数组得到一个数组
    public boolean canChoose(int[][] groups, int[] nums) {
        int m = 0;
        int n = 0;
        int i = 0;
        while (i < nums.length) {
            if (nums[i] == groups[m][n]) {
                ++n;
                if (n == groups[m].length) {
                    ++m;
                    if (m == groups.length) {
                        return true;
                    }
                    n = 0;
                }
            } else {
                i -= n;
                n = 0;
            }
            ++i;
        }
        return false;

    }

    // 1733. 需要教语言的最少人数
    public int minimumTeachings(int n, int[][] languages, int[][] friendships) {
        // // 表示不能相互沟通的好友编号
        // Set<Integer> set = new HashSet<>();
        // // key：set中每个好友掌握的每种语言种类
        // // val：每种语言的数量
        // Map<Integer, Integer> map = new HashMap<>();
        // for (int[] friendship : friendships) {
        // if (!canTalk(languages, friendship[0], friendship[1])) {
        // set.add(friendship[0]);
        // set.add(friendship[1]);
        // }
        // }
        // // 所有互为好友的学生都能互相沟通
        // if (set.isEmpty()) {
        // return 0;
        // }
        // for (int friend : set) {
        // for (int lan : languages[friend - 1]) {
        // map.put(lan, map.getOrDefault(lan, 0) + 1);
        // }
        // }

        // return set.size() - Collections.max(map.values());

        Set<Integer> set = new HashSet<>();
        for (int[] friendship : friendships) {
            if (!canTalk(languages[friendship[0] - 1], languages[friendship[1] - 1])) {
                set.add(friendship[0] - 1);
                set.add(friendship[1] - 1);
            }
        }
        if (set.isEmpty()) {
            return 0;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : set) {
            for (int lan : languages[num]) {
                map.put(lan, map.getOrDefault(lan, 0) + 1);
            }
        }
        return set.size() - Collections.max(map.values());

    }

    private boolean canTalk(int[] languages1, int[] languages2) {
        for (int lan1 : languages1) {
            for (int lan2 : languages2) {
                if (lan1 == lan2) {
                    return true;
                }
            }
        }

        return false;
    }

    // 1550. 存在连续三个奇数的数组
    public boolean threeConsecutiveOdds(int[] arr) {
        for (int i = 1; i < arr.length - 1; ++i) {
            if (arr[i - 1] % 2 == 1 && arr[i] % 2 == 1 && arr[i + 1] % 2 == 1) {
                return true;
            }
        }
        return false;

    }

    // 1550. 存在连续三个奇数的数组
    public boolean threeConsecutiveOdds2(int[] arr) {
        int count = 0;
        for (int num : arr) {
            count = num % 2 == 1 ? ++count : 0;
            if (count == 3) {
                return true;
            }
        }
        return false;

    }

    // 1471. 数组中的 k 个最强值
    public int[] getStrongest(int[] arr, int k) {
        int[] res = new int[k];
        int index = 0;
        Arrays.sort(arr);
        int m = arr[(arr.length - 1) / 2];
        int left = 0;
        int right = arr.length - 1;
        while (left <= right) {
            if (Math.abs(arr[right] - m) >= Math.abs(arr[left] - m)) {
                res[index++] = arr[right--];
            } else {
                res[index++] = arr[left++];
            }
            if (index == k) {
                return res;
            }
        }
        return res;

    }

    // 1508. 子数组和排序后的区间和
    public int rangeSum(int[] nums, int n, int left, int right) {
        final int MOD = 1000000007;
        int[] count = new int[100001];
        for (int i = 0; i < nums.length; ++i) {
            int sum = 0;
            for (int j = i; j < nums.length; ++j) {
                sum += nums[j];
                ++count[sum];
            }
        }
        --left;
        --right;
        int index = 0;
        long ans = 0;
        for (int i = 0; i < count.length; ++i) {
            for (int j = 0; j < count[i]; ++j) {
                if (index >= left && index <= right) {
                    ans = (ans + i) % MOD;
                } else if (index > right) {
                    return (int) (ans % MOD);
                }
                ++index;
            }
        }
        return (int) (ans % MOD);

    }

    // 1169. 查询无效交易
    public List<String> invalidTransactions(String[] transactions) {
        List<String> list = new ArrayList<>();
        boolean[] flag = new boolean[transactions.length];
        for (int i = 0; i < transactions.length; ++i) {
            String[] transactionI = transactions[i].split("\\,");
            if (!flag[i] && Integer.parseInt(transactionI[2]) > 1000) {
                list.add(transactions[i]);
                flag[i] = true;
            }
            for (int j = i + 1; j < transactions.length; ++j) {
                String[] transactionJ = transactions[j].split("\\,");
                if (transactionJ[0].equals(transactionI[0]) && !transactionJ[3].equals(transactionI[3])
                        && Math.abs(Integer.parseInt(transactionJ[1]) - Integer.parseInt(transactionI[1])) <= 60) {
                    if (!flag[j]) {
                        flag[j] = true;
                        list.add(transactions[j]);
                    }
                    if (!flag[i]) {
                        flag[i] = true;
                        list.add(transactions[i]);
                    }
                }
            }
        }
        return list;

    }

    // 1574. 删除最短的子数组使剩余数组有序
    public int findLengthOfShortestSubarray(int[] arr) {
        int left = 0;
        while (left < arr.length - 1) {
            if (arr[left] > arr[left + 1]) {
                break;
            }
            ++left;
        }
        if (left == arr.length - 1) {
            return 0;
        }
        int right = arr.length - 1;
        while (right >= 1) {
            if (arr[right - 1] > arr[right]) {
                break;
            }
            --right;
        }
        int res = Math.min(arr.length - left - 1, right);
        int i = 0;

        while (i <= left && right < arr.length) {
            if (arr[i] <= arr[right]) {
                res = Math.min(res, right - i - 1);
                ++i;
            } else {
                ++right;
            }
        }
        return res;

    }

    // 1503. 所有蚂蚁掉下来前的最后一刻
    public int getLastMoment(int n, int[] left, int[] right) {
        int moment = 0;
        for (int i = 0; i < left.length; ++i) {
            moment = Math.max(moment, left[i]);
        }
        for (int i = 0; i < right.length; ++i) {
            moment = Math.max(moment, n - right[i]);
        }
        return moment;

    }

    // 918. 环形子数组的最大和
    public int maxSubarraySumCircular(int[] A) {
        int max = A[0];
        int pre = 0;
        for (int i = 0; i < A.length; ++i) {
            pre = Math.max(A[i], pre + A[i]);
            max = Math.max(pre, max);
        }

        int min = 0;
        pre = 0;
        for (int i = 1; i < A.length - 1; ++i) {
            pre = Math.min(A[i], pre + A[i]);
            min = Math.min(pre, min);
        }
        return Math.max(Arrays.stream(A).sum() - min, max);

    }

    public static boolean circularArrayLoop(int[] nums) {
        boolean[] visited = new boolean[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            Arrays.fill(visited, false);
            int index = i;
            while (!visited[index] && nums[index] > 0
                    && index != ((index + nums[index]) % nums.length + nums.length) % nums.length) {
                visited[index] = true;
                index = ((index + nums[index]) % nums.length + nums.length) % nums.length;
            }
            if (visited[index]) {
                return true;
            }
        }
        for (int i = 0; i < nums.length; ++i) {
            Arrays.fill(visited, false);
            int index = i;
            while (!visited[index] && nums[index] < 0
                    && index != ((index + nums[index]) % nums.length + nums.length) % nums.length) {
                visited[index] = true;
                index = ((index + nums[index]) % nums.length + nums.length) % nums.length;
            }
            if (visited[index]) {
                return true;
            }
        }
        return false;

    }

    // 349. 两个数组的交集
    public int[] intersection(int[] nums1, int[] nums2) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums1) {
            set.add(num);
        }
        Set<Integer> set2 = new HashSet<>();
        for (int num : nums2) {
            if (set.contains(num)) {
                set2.add(num);
            }
        }
        int[] res = new int[set2.size()];
        int index = 0;
        for (int num : set2) {
            res[index++] = num;
        }
        return res;

    }

    // 349
    public int[] intersection1(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        Set<Integer> set = new HashSet<>();
        int i = 0;
        int j = 0;
        while (i < nums1.length && j < nums2.length) {
            if (nums1[i] == nums2[j]) {
                set.add(nums1[i]);
                ++i;
                ++j;
            } else if (nums1[i] < nums2[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        int[] res = new int[set.size()];
        int index = 0;
        for (int num : set) {
            res[index++] = num;
        }

        return res;
    }

    // 350. 两个数组的交集 II
    public int[] intersect(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums1) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        List<Integer> list = new ArrayList<>();
        for (int num : nums2) {
            if (map.getOrDefault(num, 0) > 0) {
                list.add(num);
                map.put(num, map.getOrDefault(num, 0) - 1);
            }
        }
        int[] res = new int[list.size()];
        int index = 0;
        for (int num : list) {
            res[index++] = num;
        }
        return res;

    }

    // 350. 两个数组的交集 II
    public int[] intersect2(int[] nums1, int[] nums2) {
        Arrays.sort(nums1);
        Arrays.sort(nums2);
        int i = 0;
        int j = 0;
        List<Integer> list = new ArrayList<>();
        while (i < nums1.length && j < nums2.length) {
            if (nums1[i] == nums2[j]) {
                list.add(nums1[i]);
                ++i;
                ++j;
            } else if (nums1[i] < nums2[j]) {
                ++i;
            } else {
                ++j;
            }
        }
        int[] res = new int[list.size()];
        int index = 0;
        for (int num : list) {
            res[index++] = num;
        }
        return res;

    }

    // 7. 整数反转
    public int reverse(int x) {
        int res = 0;
        while (x != 0) {
            int mod = x % 10;
            x /= 10;
            if (res > Integer.MAX_VALUE / 10 || res < Integer.MIN_VALUE / 10) {
                return 0;
            }
            res = res * 10 + mod;
        }
        return res;

    }

    // 面试题 10.05. 稀疏数组搜索
    public int findString(String[] words, String s) {

        int left = 0;
        int right = words.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if ("".equals(words[mid]) && !s.equals(words[left])) {
                ++left;
                continue;
            } else if (s.equals(words[left])) {
                return left;
            }

            if (words[mid].compareTo(s) == 0) {
                return mid;
            } else if (words[mid].compareTo(s) > 0) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;
    }

    // 392. 判断子序列
    public boolean isSubsequence(String s, String t) {
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

    // 1800. 最大升序子数组和
    public int maxAscendingSum(int[] nums) {
        int max = nums[0];
        int cur = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] > nums[i - 1]) {
                cur += nums[i];
            } else {
                max = Math.max(max, cur);
                cur = nums[i];
            }
        }
        return Math.max(max, cur);

    }

    // 1854. 人口最多的年份--模拟法
    public int maximumPopulation(int[][] logs) {
        int[] counts = new int[101];
        for (int[] log : logs) {
            for (int i = log[0] - 1950; i < log[1] - 1950; ++i) {
                ++counts[i];
            }
        }
        int max = Arrays.stream(counts).max().getAsInt();
        for (int i = 0; i < counts.length; ++i) {
            if (max == counts[i]) {
                return i + 1950;
            }
        }
        return -1;

    }

    // 1854. 人口最多的年份--差分数组
    public int maximumPopulation2(int[][] logs) {
        int[] diff = new int[101];
        for (int[] log : logs) {
            ++diff[log[0] - 1950];
            --diff[log[1] - 1950];
        }
        int max = diff[0];
        int index = 0;
        for (int i = 1; i < diff.length; ++i) {
            diff[i] += diff[i - 1];
            if (diff[i] > max) {
                max = diff[i];
                index = i;
            }
        }
        return index + 1950;

    }

    // 1094. 拼车
    public boolean carPooling(int[][] trips, int capacity) {
        int[] diff = new int[1001];
        for (int[] trip : trips) {
            diff[trip[1]] += trip[0];
            diff[trip[2]] -= trip[0];
        }
        int max = diff[0];
        for (int i = 1; i < diff.length; ++i) {
            diff[i] += diff[i - 1];
            max = Math.max(max, diff[i]);
            if (max > capacity) {
                return false;
            }
        }
        return true;

    }

    // 1848. 到目标元素的最小距离
    public int getMinDistance(int[] nums, int target, int start) {
        int res = Integer.MAX_VALUE;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == target) {
                res = Math.min(res, Math.abs(i - start));
            }
        }
        return res;

    }

    // 1861. 旋转盒子
    public char[][] rotateTheBox(char[][] box) {
        char[][] res = new char[box[0].length][box.length];
        for (int i = 0; i < box.length; ++i) {
            int pos = box[i].length - 1;
            for (int j = box[i].length - 1; j >= 0; --j) {
                if (box[i][j] == '*') {
                    pos = j - 1;
                } else if (box[i][j] == '#') {
                    box[i][pos--] = '#';
                    if (pos != j - 1) {
                        box[i][j] = '.';
                    }
                }
            }
        }
        for (int i = 0; i < box.length; ++i) {
            for (int j = 0; j < box[0].length; ++j) {
                res[j][box.length - i - 1] = box[i][j];
            }
        }
        return res;

    }

    // 367. 有效的完全平方数
    public boolean isPerfectSquare(int num) {
        if (num == 1) {
            return true;
        }
        long left = 1;
        long right = num / 2;
        while (left <= right) {
            long mid = left + ((right - left) >>> 1);
            if (mid * mid == (long) num) {
                return true;
            } else if (mid * mid > num) {
                right = mid - 1;
            } else if (mid * mid < num) {
                left = mid + 1;
            }
        }
        return false;
    }

    // 744. 寻找比目标字母大的最小字母
    public char nextGreatestLetter(char[] letters, char target) {
        if (target < letters[0] || target >= letters[letters.length - 1]) {
            return letters[0];
        }
        char ans = 'a';
        int left = 0;
        int right = letters.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (letters[mid] > target) {
                ans = letters[mid];
                right = mid - 1;
            } else if (letters[mid] <= target) {
                left = mid + 1;
            }
        }
        return ans;

    }

    // 面试题 10.09. 排序矩阵查找
    public boolean searchMatrix10_09(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return false;
        }
        int j = matrix[0].length - 1;
        int i = 0;
        while (i < matrix.length && j >= 0) {
            if (matrix[i][j] == target) {
                return true;
            } else if (matrix[i][j] > target) {
                --j;
            } else if (matrix[i][j] < target) {
                ++i;
            }
        }
        return false;

    }

    // 1855. 下标对中的最大距离
    public int maxDistance(int[] nums1, int[] nums2) {
        int res = 0;
        int i = nums1.length - 1;
        for (int j = nums2.length - 1; j >= 0; --j) {
            i = Math.min(i, j);
            while (i >= 0) {
                if (nums1[i] <= nums2[j]) {
                    res = Math.max(res, j - i);
                    --i;
                } else {
                    break;
                }
            }
        }
        return res;

    }

    // 378. 有序矩阵中第 K 小的元素
    public int kthSmallest(int[][] matrix, int k) {
        int ans = 0;
        int left = matrix[0][0];
        int right = matrix[matrix.length - 1][matrix[0].length - 1];
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check378(mid, matrix) < k) {
                left = mid + 1;
            } else {
                ans = mid;
                right = mid - 1;
            }
        }
        return ans;

    }

    private int check378(int num, int[][] matrix) {
        int count = 0;
        int i = matrix.length - 1;
        int j = 0;
        while (i >= 0 && j < matrix[0].length) {
            if (matrix[i][j] <= num) {
                count += i + 1;
                ++j;
            } else {
                --i;
            }

        }
        return count;
    }

    // 454. 四数相加 II
    public int fourSumCount(int[] nums1, int[] nums2, int[] nums3, int[] nums4) {
        int count = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num1 : nums1) {
            for (int num2 : nums2) {
                map.put(num1 + num2, map.getOrDefault(num1 + num2, 0) + 1);
            }
        }
        for (int num3 : nums3) {
            for (int num4 : nums4) {
                count += map.getOrDefault(-num3 - num4, 0);
            }
        }
        return count;
    }

    // 1202. 交换字符串中的元素
    public String smallestStringWithSwaps(String s, List<List<Integer>> pairs) {
        Union1202 union = new Union1202(s.length());
        for (List<Integer> pair : pairs) {
            union.union(pair.get(0), pair.get(1));
        }
        Map<Integer, List<Character>> map = new HashMap<>();
        for (int i = 0; i < s.length(); ++i) {
            int root = union.getRoot(i);
            map.computeIfAbsent(root, k -> new LinkedList<>()).add(s.charAt(i));
        }
        for (List<Character> list : map.values()) {
            Collections.sort(list, new Comparator<Character>() {
                @Override
                public int compare(Character o1, Character o2) {
                    return o1 - o2;
                }
            });
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < s.length(); ++i) {
            int root = union.getRoot(i);
            List<Character> list = map.get(root);
            res.append(list.remove(0));
        }
        return res.toString();

    }

    public class Union1202 {
        private int[] parent;
        private int[] rank;

        public Union1202(int n) {
            parent = new int[n];
            rank = new int[n];
            Arrays.fill(rank, 1);
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

    // 778. 水位上升的泳池中游泳
    public int swimInWater(int[][] grid) {
        int n = grid.length;
        Union778 union = new Union778(n * n);
        int[] index = new int[n * n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                index[grid[i][j]] = getIndex778(n, i, j);
            }
        }
        for (int time = 0; time < n * n; ++time) {
            int x = index[time] / n;
            int y = index[time] % n;
            if (x - 1 >= 0 && grid[x - 1][y] < time) {
                union.union(getIndex778(n, x, y), getIndex778(n, x - 1, y));
            }
            if (y - 1 >= 0 && grid[x][y - 1] < time) {
                union.union(getIndex778(n, x, y), getIndex778(n, x, y - 1));
            }
            if (x + 1 < n && grid[x + 1][y] < time) {
                union.union(getIndex778(n, x, y), getIndex778(n, x + 1, y));
            }
            if (y + 1 < n && grid[x][y + 1] < time) {
                union.union(getIndex778(n, x, y), getIndex778(n, x, y + 1));
            }
            if (union.isConnected(0, n * n - 1)) {
                return time;
            }
        }
        return -1;

    }

    private int getIndex778(int n, int i, int j) {
        return n * i + j;
    }

    public class Union778 {
        private int[] parent;
        private int[] rank;

        public Union778(int n) {
            parent = new int[n];
            rank = new int[n];
            Arrays.fill(rank, 1);
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

    // 200. 岛屿数量
    public int numIslands(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Union200 union = new Union200(grid);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == '1') {
                    if (i - 1 >= 0 && grid[i - 1][j] == '1') {
                        union.union(getIndex200(n, i, j), getIndex200(n, i - 1, j));
                    }
                    if (j - 1 >= 0 && grid[i][j - 1] == '1') {
                        union.union(getIndex200(n, i, j), getIndex200(n, i, j - 1));
                    }
                }
            }
        }
        return union.getCount();

    }

    private int getIndex200(int n, int i, int j) {
        return n * i + j;
    }

    public class Union200 {
        private int[] parent;
        private int[] rank;
        private int count;

        public Union200(char[][] grid) {
            int m = grid.length;
            int n = grid[0].length;
            parent = new int[m * n];
            rank = new int[m * n];
            Arrays.fill(rank, 1);
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (grid[i][j] == '1') {
                        parent[i * n + j] = i * n + j;
                        ++count;
                    }
                }
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
            --count;
        }

        public int getCount() {
            return count;
        }
    }

    // 1319. 连通网络的操作次数
    public int makeConnected(int n, int[][] connections) {
        if (connections.length < n - 1) {
            return -1;
        }
        Union1319 union = new Union1319(n);
        for (int[] connection : connections) {
            union.union(connection[0], connection[1]);
        }
        return union.getCount() - 1;

    }

    public class Union1319 {
        private int[] parent;
        private int[] rank;
        private int count;

        public Union1319(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            rank = new int[n];
            Arrays.fill(rank, 1);
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

    // 684. 冗余连接
    public int[] findRedundantConnection(int[][] edges) {
        Union684 union = new Union684(edges.length);
        for (int[] edge : edges) {
            if (union.isConnected(edge[0] - 1, edge[1] - 1)) {
                return edge;
            }
            union.union(edge[0] - 1, edge[1] - 1);
        }
        return null;

    }

    public class Union684 {
        private int[] parent;
        private int[] rank;

        public Union684(int n) {
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

    // 547. 省份数量
    public int findCircleNum(int[][] isConnected) {
        int n = isConnected.length;
        Union547 union = new Union547(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j && isConnected[i][j] == 1) {
                    union.union(i, j);
                }
            }
        }
        return union.getCount();

    }

    public class Union547 {
        private int[] rank;
        private int[] parent;
        private int count;

        public Union547(int n) {
            parent = new int[n];
            rank = new int[n];
            Arrays.fill(rank, 1);
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

    // 959. 由斜杠划分区域
    public int regionsBySlashes(String[] grid) {
        int n = grid.length;
        Union959 union = new Union959(n * n * 4);
        int index = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                char c = grid[i].charAt(j);
                // 方格内合并
                if (c == ' ') {
                    union.union(index, index + 1);
                    union.union(index + 1, index + 2);
                    union.union(index + 2, index + 3);
                } else if (c == '/') {
                    union.union(index, index + 1);
                    union.union(index + 2, index + 3);
                } else if (c == '\\') {
                    union.union(index + 1, index + 2);
                    union.union(index, index + 3);
                }

                // 方格间合并
                if (j + 1 < n) {
                    union.union(index + 3, index + 5);
                }
                if (i + 1 < n) {
                    union.union(index + 2, index + n * 4);
                }
                index += 4;
            }
        }
        return union.getCount();

    }

    public class Union959 {
        private int[] parent;
        private int[] rank;
        private int count;

        public Union959(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            rank = new int[n];
            Arrays.fill(rank, 1);
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

    // 765. 情侣牵手
    public int minSwapsCouples(int[] row) {
        Union765 union = new Union765(row.length / 2);
        for (int i = 0; i < row.length; i += 2) {
            union.union(row[i] / 2, row[i + 1] / 2);
        }
        return row.length / 2 - union.getCount();

    }

    public class Union765 {
        private int[] rank;
        private int[] parent;
        private int count;

        public Union765(int n) {
            parent = new int[n];
            rank = new int[n];
            Arrays.fill(rank, 1);
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
            if (rank[root1] < rank[root2]) {
                parent[root1] = root2;
            } else if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
            } else {
                parent[root1] = root2;
                ++rank[root2];
            }
            --count;

        }

        public int getCount() {
            return count;
        }

    }

    // 947. 移除最多的同行或同列石头
    public int removeStones(int[][] stones) {

        int index = 0;
        Map<Integer, Integer> map = new HashMap<>();

        for (int[] stone : stones) {
            if (!map.containsKey(stone[0])) {
                map.put(stone[0], index++);
            }
            if (!map.containsKey(stone[1])) {
                map.put(stone[1], index++);
            }

        }
        Union947 union947 = new Union947(20001, index);
        for (int[] stone : stones) {
            union947.union(stone[0], stone[1] + 10000);
        }

        return stones.length - union947.getCount();

    }

    public class Union947 {
        private int[] rank;
        private int[] parent;
        private int count;

        public Union947(int n, int count) {
            parent = new int[n];
            rank = new int[n];
            Arrays.fill(rank, 1);
            for (int i = 0; i < parent.length; ++i) {
                parent[i] = i;
            }
            this.count = count;
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
            } else if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
            } else {
                parent[root1] = root2;
                ++rank[root2];
            }
            --count;
        }

        public int getCount() {
            return count;
        }

    }

    // 839. 相似字符串组
    public int numSimilarGroups(String[] strs) {
        Union839 union = new Union839(strs.length);
        for (int i = 0; i < strs.length; ++i) {
            for (int j = i + 1; j < strs.length; ++j) {
                if (union.isConnected(i, j)) {
                    continue;
                }
                if (isSimilar(strs[i], strs[j])) {
                    union.union(i, j);
                }
            }
        }
        return union.getCount();

    }

    private boolean isSimilar(String s1, String s2) {
        int count = 0;
        for (int i = 0; i < s1.length(); ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                if (++count > 2) {
                    return false;
                }
            }
        }
        return true;
    }

    public class Union839 {
        private int[] rank;
        private int[] parent;
        private int count;

        public Union839(int n) {
            rank = new int[n];
            parent = new int[n];
            Arrays.fill(rank, 1);
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

    // 990. 等式方程的可满足性
    public boolean equationsPossible(String[] equations) {
        Union990 union = new Union990();
        for (String equation : equations) {
            if (equation.charAt(1) == '=') {
                union.union(equation.charAt(0) - 'a', equation.charAt(3) - 'a');
            }
        }
        for (String equation : equations) {
            if (equation.charAt(1) == '!') {
                if (union.isConnected(equation.charAt(0) - 'a', equation.charAt(3) - 'a')) {
                    return false;
                }
            }
        }
        return true;

    }

    public class Union990 {
        private int[] parent;
        private int[] rank;

        public Union990() {
            parent = new int[26];
            rank = new int[26];
            Arrays.fill(rank, 1);
            for (int i = 0; i < parent.length; ++i) {
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

    // 1579. 保证图可完全遍历
    public int maxNumEdgesToRemove(int n, int[][] edges) {
        int res = 0;
        Union1579 unionAlice = new Union1579(n);
        Union1579 unionBob = new Union1579(n);
        for (int[] edge : edges) {
            // 共用边
            if (edge[0] == 3) {
                if (!unionAlice.isConnected(edge[1] - 1, edge[2] - 1)) {
                    unionAlice.union(edge[1] - 1, edge[2] - 1);
                    unionBob.union(edge[1] - 1, edge[2] - 1);
                } else {
                    ++res;
                }
            }
        }
        for (int[] edge : edges) {
            // Alice
            if (edge[0] == 1) {
                if (!unionAlice.isConnected(edge[1] - 1, edge[2] - 1)) {
                    unionAlice.union(edge[1] - 1, edge[2] - 1);
                } else {
                    ++res;
                }
            }
            // Bob
            else if (edge[0] == 2) {
                if (!unionBob.isConnected(edge[1] - 1, edge[2] - 1)) {
                    unionBob.union(edge[1] - 1, edge[2] - 1);
                } else {
                    ++res;
                }
            }
        }
        if (unionAlice.getCount() != 1 || unionBob.getCount() != 1) {
            return -1;
        }
        return res;

    }

    public class Union1579 {
        private int[] parent;
        private int[] rank;
        private int count;

        public Union1579(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            rank = new int[n];
            Arrays.fill(rank, 1);
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

    // 1631. 最小体力消耗路径
    public int minimumEffortPath(int[][] heights) {
        int rows = heights.length;
        int cols = heights[0].length;
        Union1631 union;
        int left = 0;
        int right = 1000000;
        int ans = 0;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            union = new Union1631(rows * cols);
            if (check(union, heights, mid)) {
                ans = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return ans;

    }

    private boolean check(Union1631 union, int[][] heights, int mid) {
        int rows = heights.length;
        int cols = heights[0].length;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (i + 1 < rows && Math.abs(heights[i][j] - heights[i + 1][j]) <= mid) {
                    union.union(i * cols + j, (i + 1) * cols + j);
                }
                if (j + 1 < cols && Math.abs(heights[i][j] - heights[i][j + 1]) <= mid) {
                    union.union(i * cols + j, i * cols + (j + 1));
                }
                if (union.isConnected(0, cols * rows - 1)) {
                    return true;
                }
            }
        }
        return false;
    }

    // 1631. 最小体力消耗路径
    public int minimumEffortPath2(int[][] heights) {
        int m = heights.length;
        int n = heights[0].length;
        Union1631 union = new Union1631(m * n);
        List<int[]> list = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i + 1 < m) {
                    list.add(new int[] { getIndex1631(n, i, j), getIndex1631(n, i + 1, j),
                            Math.abs(heights[i][j] - heights[i + 1][j]) });
                }
                if (j + 1 < n) {
                    list.add(new int[] { getIndex1631(n, i, j), getIndex1631(n, i, j + 1),
                            Math.abs(heights[i][j] - heights[i][j + 1]) });
                }
            }
        }
        Collections.sort(list, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }
        });
        for (int[] item : list) {
            union.union(item[0], item[1]);
            if (union.isConnected(0, m * n - 1)) {
                return item[2];
            }
        }
        return 0;

    }

    private int getIndex1631(int n, int i, int j) {
        return i * n + j;
    }

    public class Union1631 {
        private int[] parent;
        private int[] rank;

        public Union1631(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            rank = new int[n];
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
        }

    }

    // 1722. 执行交换操作后的最小汉明距离
    public int minimumHammingDistance(int[] source, int[] target, int[][] allowedSwaps) {
        int res = 0;
        Union1722 union = new Union1722(source.length);

        for (int[] allowedSwap : allowedSwaps) {
            union.union(allowedSwap[0], allowedSwap[1]);
        }
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < source.length; ++i) {
            map.computeIfAbsent(target[i], k -> new LinkedList<>()).add(i);
        }
        for (int i = 0; i < source.length; ++i) {
            if (!map.containsKey(source[i])) {
                ++res;
                continue;
            }
            List<Integer> list = map.get(source[i]);
            Iterator<Integer> iterator = list.iterator();
            boolean flag = false;
            while (iterator.hasNext()) {
                Integer index = iterator.next();
                if (union.isConnected(i, index)) {
                    iterator.remove();
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                ++res;

            }
        }

        return res;

    }

    public class Union1722 {
        private int[] rank;
        private int[] parent;

        public Union1722(int n) {
            rank = new int[n];
            parent = new int[n];
            Arrays.fill(rank, 1);
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
            } else if (rank[root1] > rank[root2]) {
                parent[root2] = root1;
            } else {
                parent[root1] = root2;
                ++rank[root2];
            }
        }

    }

    // 130. 被围绕的区域
    public void solve(char[][] board) {
        int m = board.length;
        int n = board[0].length;
        Union130 union = new Union130(m * n + 1);
        int dummy = m * n;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 'O') {
                    if (i == 0 || j == 0 || i == m - 1 || j == n - 1) {
                        union.union(getIndex130(n, i, j), dummy);
                    } else {
                        if (board[i - 1][j] == 'O') {
                            union.union(getIndex130(n, i, j), getIndex130(n, i - 1, j));
                        }
                        if (board[i + 1][j] == 'O') {
                            union.union(getIndex130(n, i, j), getIndex130(n, i + 1, j));
                        }
                        if (board[i][j - 1] == 'O') {
                            union.union(getIndex130(n, i, j), getIndex130(n, i, j - 1));
                        }
                        if (board[i][j + 1] == 'O') {
                            union.union(getIndex130(n, i, j), getIndex130(n, i, j + 1));
                        }
                    }
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 'O' && !union.isConnected(getIndex130(n, i, j), dummy)) {
                    board[i][j] = 'X';
                }
            }
        }
    }

    private int getIndex130(int n, int i, int j) {
        return i * n + j;
    }

    public class Union130 {
        private int[] parent;
        private int[] rank;

        public Union130(int n) {
            parent = new int[n];
            rank = new int[n];
            Arrays.fill(rank, 1);
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

    // 721. 账户合并
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        Map<String, Integer> emailToIndex = new HashMap<>();
        Map<String, String> emailToName = new HashMap<>();
        int index = 0;
        for (List<String> account : accounts) {

            for (int i = 1; i < account.size(); ++i) {
                if (!emailToIndex.containsKey(account.get(i))) {
                    emailToIndex.put(account.get(i), index++);
                    emailToName.put(account.get(i), account.get(0));
                }
            }
        }
        Union721 union = new Union721(index);
        for (List<String> account : accounts) {
            int index1 = emailToIndex.get(account.get(1));
            for (int i = 2; i < account.size(); ++i) {
                union.union(index1, emailToIndex.get(account.get(i)));
            }
        }
        Map<Integer, List<String>> mergedAccounts = new HashMap<>();
        for (Map.Entry<String, Integer> entry : emailToIndex.entrySet()) {
            String key = entry.getKey();
            int value = entry.getValue();
            int root = union.getRoot(value);
            mergedAccounts.computeIfAbsent(root, k -> new ArrayList<>()).add(key);
        }
        List<List<String>> res = new ArrayList<>();
        for (List<String> list : mergedAccounts.values()) {
            Collections.sort(list);
            String name = emailToName.get(list.get(0));
            List<String> sub = new ArrayList<>();
            sub.add(name);
            sub.addAll(list);
            res.add(sub);
        }
        return res;

    }

    public class Union721 {
        private int[] parent;
        private int[] rank;

        public Union721(int n) {
            parent = new int[n];
            rank = new int[n];
            Arrays.fill(rank, 1);
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

    // 399. 除法求值
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

    // 1869. 哪种连续子字符串更长
    public static boolean checkZeroOnes(String s) {
        int maxZero = 0;
        int curZero = 0;
        int maxOne = 0;
        int curOne = 0;
        for (char a : s.toCharArray()) {
            if (a - '0' == 0) {
                maxOne = Math.max(maxOne, curOne);
                curOne = 0;
                ++curZero;
            } else {
                maxZero = Math.max(maxZero, curZero);
                curZero = 0;
                ++curOne;
            }
        }
        maxZero = Math.max(maxZero, curZero);
        maxOne = Math.max(maxOne, curOne);
        return maxOne > maxZero;

    }

    // 1249. 移除无效的括号
    public String minRemoveToMakeValid(String s) {
        Stack<Integer> stack = new Stack<>();
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else if (s.charAt(i) == ')') {
                if (stack.isEmpty()) {
                    set.add(i);
                } else {
                    stack.pop();
                }
            }
        }
        while (!stack.isEmpty()) {
            set.add(stack.pop());
        }

        StringBuilder res = new StringBuilder();
        for (int i = 0; i < s.length(); ++i) {
            if (!set.contains(i)) {
                res.append(s.charAt(i));
            }
        }
        return res.toString();

    }

    // 1249. 移除无效的括号
    public String minRemoveToMakeValid2(String s) {
        StringBuilder res = new StringBuilder(s);
        makeValid1249(res, '(', ')');
        makeValid1249(res.reverse(), ')', '(');
        return res.reverse().toString();

    }

    private void makeValid1249(StringBuilder builder, char open, char close) {
        int count = 0;
        for (int i = 0; i < builder.length(); ++i) {
            if (builder.charAt(i) == open) {
                ++count;
            } else if (builder.charAt(i) == close) {
                if (count > 0) {
                    --count;
                } else {
                    builder.deleteCharAt(i);
                    --i;
                }
            }
        }

    }

    // 1249. 移除无效的括号
    public String minRemoveToMakeValid3(String s) {
        int balance = 0;
        int seen = 0;
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                ++balance;
                ++seen;
            } else if (s.charAt(i) == ')') {
                if (balance > 0) {
                    --balance;
                } else {
                    continue;
                }
            }
            builder.append(s.charAt(i));
        }
        int openToKeep = seen - balance;
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < builder.length(); ++i) {
            if (builder.charAt(i) == '(') {
                if (openToKeep > 0) {
                    --openToKeep;
                } else {
                    continue;
                }
            }
            result.append(builder.charAt(i));
        }
        return result.toString();

    }

    // 1541. 平衡括号字符串的最少插入次数
    public int minInsertions(String s) {
        int left = 0;
        int count = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                ++left;
            } else {
                if (left > 0) {
                    --left;
                } else {
                    ++count;
                }
                if (i + 1 == s.length() || s.charAt(i + 1) == '(') {
                    ++count;
                } else {
                    ++i;
                }
            }
        }
        return left * 2 + count;

    }

    // 1209. 删除字符串中的所有相邻重复项 II -- 超时
    public String removeDuplicates1209(String s, int k) {
        StringBuilder builder = new StringBuilder(s);
        int length = -1;
        while (length != builder.length()) {
            length = builder.length();
            int count = 1;
            for (int i = 0; i < builder.length(); ++i) {
                if (i == 0 || builder.charAt(i) != builder.charAt(i - 1)) {
                    count = 1;
                } else if (++count == k) {
                    builder.delete(i - k + 1, i + 1);
                    break;
                }
            }
        }
        return builder.toString();

    }

    // 1209. 删除字符串中的所有相邻重复项 II
    public String removeDuplicates1209_2(String s, int k) {
        StringBuilder res = new StringBuilder(s);
        int[] counts = new int[s.length()];
        for (int i = 0; i < res.length(); ++i) {
            if (i == 0 || res.charAt(i) != res.charAt(i - 1)) {
                counts[i] = 1;
            } else {
                counts[i] = counts[i - 1] + 1;
                if (counts[i] == k) {
                    res.delete(i - k + 1, i + 1);
                    i -= k;
                }
            }
        }
        return res.toString();

    }

    // 1209. 删除字符串中的所有相邻重复项 II
    public String removeDuplicates1209_3(String s, int k) {
        StringBuilder res = new StringBuilder(s);
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < res.length(); ++i) {
            if (i == 0 || res.charAt(i) != res.charAt(i - 1)) {
                stack.push(1);
            } else {
                int count = stack.pop() + 1;
                if (count == k) {
                    res.delete(i - k + 1, i + 1);
                    i -= k;
                } else {
                    stack.push(count);
                }

            }
        }
        return res.toString();

    }

    // 1544. 整理字符串
    public String makeGood(String s) {
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (!stack.isEmpty() && Math.abs(c - stack.peek()) == 32) {
                stack.pop();
            } else {
                stack.push(c);
            }
        }
        StringBuilder builder = new StringBuilder();
        while (!stack.isEmpty()) {
            builder.append(stack.pop());
        }
        return builder.reverse().toString();

    }

    // 1544. 整理字符串
    public String makeGood2(String s) {
        StringBuilder builder = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (builder.length() != 0 && Math.abs(c - builder.charAt(builder.length() - 1)) == 32) {
                builder.deleteCharAt(builder.length() - 1);
            } else {
                builder.append(c);
            }
        }
        return builder.toString();

    }

    // 1190. 反转每对括号间的子串
    // "(ed(et(oc))el)"
    public String reverseParentheses(String s) {
        Stack<Integer> stack = new Stack<>();
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; ++i) {
            if (chars[i] == '(') {
                stack.push(i);
            } else if (chars[i] == ')') {
                reverse1190(chars, stack.pop(), i);
            }
        }
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < chars.length; ++i) {
            if (chars[i] != '(' && chars[i] != ')') {
                res.append(chars[i]);
            }
        }
        return res.toString();

    }

    // 1190. 反转每对括号间的子串
    // "(ed(et(oc))el)"
    public String reverseParentheses2(String s) {
        int[] pair = new int[s.length()];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else if (s.charAt(i) == ')') {
                int j = stack.pop();
                pair[j] = i;
                pair[i] = j;
            }
        }
        int sign = 1;
        int index = 0;
        StringBuilder res = new StringBuilder();
        while (index < s.length()) {
            if (s.charAt(index) == '(' || s.charAt(index) == ')') {
                sign *= -1;
                index = pair[index];
            } else {
                res.append(s.charAt(index));
            }
            index += sign;
        }
        return res.toString();

    }

    private void reverse1190(char[] chars, int i, int j) {
        while (i < j) {
            char temp = chars[i];
            chars[i] = chars[j];
            chars[j] = temp;
            ++i;
            --j;
        }
    }

    // 739. 每日温度
    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[temperatures.length];
        for (int i = 0; i < temperatures.length; ++i) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int index = stack.pop();
                res[index] = i - index;
            }
            stack.push(i);
        }
        return res;

    }

    // 856. 括号的分数
    public int scoreOfParentheses(String s) {
        Stack<Integer> stack = new Stack<>();
        stack.push(0);
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                stack.push(0);
            } else {
                int last = stack.pop();
                int secondTolast = stack.pop();
                stack.push(secondTolast + Math.max(2 * last, 1));

            }
        }
        return stack.pop();

    }

    // 856. 括号的分数
    public int scoreOfParentheses2(String s) {
        int left = 0;
        int score = 0;
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                ++left;
            } else {
                --left;
                if (s.charAt(i - 1) == '(') {
                    score += 1 << left;
                }
            }
        }
        return score;

    }

    // 901. 股票价格跨度
    class StockSpanner {
        Stack<Integer> stack;
        Stack<Integer> weight;

        public StockSpanner() {
            stack = new Stack<>();
            weight = new Stack<>();
        }

        public int next(int price) {
            int w = 1;
            while (!stack.isEmpty() && stack.peek() <= price) {
                stack.pop();
                w += weight.pop();
            }
            stack.push(price);
            weight.push(w);
            return w;
        }
    }

    // 921. 使括号有效的最少添加
    public int minAddToMakeValid(String s) {
        int count = 0;
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                if (stack.isEmpty()) {
                    ++count;
                } else {
                    stack.pop();
                }
            }
        }
        return stack.size() + count;
    }

    // 921. 使括号有效的最少添加
    public int minAddToMakeValid2(String s) {
        int left = 0;
        int count = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                ++left;
            } else {
                if (left == 0) {
                    ++count;
                } else {
                    --left;
                }
            }
        }
        return count + left;

    }

    // 394. 字符串解码
    private static int ptr = 0;

    public static String decodeString(String s) {
        Stack<String> stack = new Stack<>();
        while (ptr < s.length()) {
            char a = s.charAt(ptr);
            if (Character.isDigit(a)) {
                String num = getNum(s);
                stack.push(num);
            } else if (Character.isLetter(a) || a == '[') {
                stack.push(String.valueOf(a));
                ++ptr;
            } else {
                ++ptr;
                List<String> sub = new LinkedList<>();
                while (!stack.peek().equals("[")) {
                    sub.add(stack.pop());
                }
                Collections.reverse(sub);
                stack.pop();
                int times = Integer.parseInt(stack.pop());
                StringBuilder subTimes = new StringBuilder();
                while (times > 0) {
                    --times;
                    subTimes.append(getString(sub));
                }
                stack.push(subTimes.toString());
            }
        }
        StringBuilder res = new StringBuilder();
        for (String item : stack) {
            res.append(item);
        }
        return res.toString();

    }

    private static String getString(List<String> sub) {
        StringBuilder builder = new StringBuilder();
        for (String s : sub) {
            builder.append(s);
        }
        return builder.toString();
    }

    private static String getNum(String s) {
        StringBuilder builder = new StringBuilder();
        while (Character.isDigit(s.charAt(ptr))) {
            builder.append(s.charAt(ptr));
            ++ptr;
        }
        return builder.toString();
    }

    // 1441. 用栈操作构建数组
    public List<String> buildArray(int[] target, int n) {
        List<String> res = new ArrayList<>();
        int index = 0;
        int num = 1;
        while (index < target.length) {
            while (num++ != target[index]) {
                res.add("Push");
                res.add("Pop");
            }
            res.add("Push");
            ++index;
        }
        return res;

    }

    // 682. 棒球比赛
    public int calPoints(String[] ops) {
        Stack<Integer> stack = new Stack<>();
        for (String op : ops) {
            if ("C".equals(op)) {
                stack.pop();
            } else if ("D".equals(op)) {
                stack.push(Integer.parseInt(String.valueOf(stack.peek())) * 2);
            } else if ("+".equals(op)) {
                int last = stack.pop();
                int push = last + stack.peek();
                stack.push(last);
                stack.push(push);
            } else {
                stack.push(Integer.parseInt(op));
            }
        }
        int res = 0;
        while (!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;

    }

    // 844. 比较含退格的字符串
    public boolean backspaceCompare(String s, String t) {
        Stack<Character> sStack = new Stack<>();
        for (char sChar : s.toCharArray()) {
            if (sChar != '#') {
                sStack.push(sChar);
            } else {
                if (!sStack.isEmpty()) {
                    sStack.pop();
                }
            }
        }
        Stack<Character> tStack = new Stack<>();
        for (char tChar : t.toCharArray()) {
            if (tChar != '#') {
                tStack.push(tChar);
            } else {
                if (!tStack.isEmpty()) {
                    tStack.pop();
                }
            }
        }
        while (!sStack.isEmpty() && !tStack.isEmpty()) {
            if (sStack.pop() != tStack.pop()) {
                return false;
            }
        }
        return sStack.isEmpty() && tStack.isEmpty();

    }

    // 844. 比较含退格的字符串
    public boolean backspaceCompare2(String s, String t) {
        StringBuilder sBuilder = new StringBuilder();
        for (char a : s.toCharArray()) {
            if (a != '#') {
                sBuilder.append(a);
            } else {
                if (sBuilder.length() != 0) {
                    sBuilder.deleteCharAt(sBuilder.length() - 1);
                }
            }
        }
        StringBuilder tBuilder = new StringBuilder();
        for (char a : t.toCharArray()) {
            if (a != '#') {
                tBuilder.append(a);
            } else {
                if (tBuilder.length() != 0) {
                    tBuilder.deleteCharAt(tBuilder.length() - 1);
                }
            }
        }
        return sBuilder.toString().equals(tBuilder.toString());

    }

    // 844. 比较含退格的字符串
    public boolean backspaceCompare3(String s, String t) {
        int sIndex = s.length() - 1;
        int sSkip = 0;
        int tIndex = t.length() - 1;
        int tSkip = 0;
        while (sIndex >= 0 || tIndex >= 0) {
            while (sIndex >= 0) {
                if (s.charAt(sIndex) == '#') {
                    ++sSkip;
                    --sIndex;
                } else if (sSkip > 0) {
                    --sIndex;
                    --sSkip;
                } else {
                    break;
                }
            }
            while (tIndex >= 0) {
                if (t.charAt(tIndex) == '#') {
                    ++tSkip;
                    --tIndex;
                } else if (tSkip > 0) {
                    --tIndex;
                    --tSkip;
                } else {
                    break;
                }
            }
            if (sIndex >= 0 && tIndex >= 0) {
                if (s.charAt(sIndex) != t.charAt(tIndex)) {
                    return false;
                }
            } else if (sIndex >= 0 || tIndex >= 0) {
                return false;
            }
            --sIndex;
            --tIndex;
        }
        return true;
    }

    // 224. 基本计算器
    // s =
    // "1-(3+5-2+(3+19-(3-1-4+(9-4-(4-(1+(3)-2)-5)+8-(3-5)-1)-4)-5)-4+3-9)-4-(3+2-5)-10"
    public int calculate224(String s) {
        Stack<Integer> stack = new Stack<>();
        stack.push(1);
        int sign = 1;
        int i = 0;
        int res = 0;
        while (i < s.length()) {
            if (Character.isWhitespace(s.charAt(i))) {
                ++i;
            } else if (s.charAt(i) == '+') {
                sign = stack.peek();
                ++i;
            } else if (s.charAt(i) == '-') {
                sign = -stack.peek();
                ++i;
            } else if (s.charAt(i) == '(') {
                stack.push(sign);
                ++i;
            } else if (s.charAt(i) == ')') {
                stack.pop();
                ++i;
            } else {
                long temp = 0;
                while (i < s.length() && Character.isDigit(s.charAt(i))) {
                    temp = temp * 10 + s.charAt(i) - '0';
                    ++i;
                }
                res += temp * sign;
            }
        }
        return res;

    }

    // 946. 验证栈序列
    // pushed = [1,2,3,4,5], popped = [4,5,3,2,1] true
    // pushed = [1,2,3,4,5], popped = [4,3,5,1,2] false
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();
        int j = 0;
        for (int push : pushed) {
            stack.push(push);
            while (!stack.isEmpty() && popped[j] == stack.peek()) {
                stack.pop();
                ++j;
            }
        }
        return j == popped.length;

    }

    // 1381. 设计一个支持增量操作的栈
    class CustomStack {
        private int index;
        private int[] array;

        public CustomStack(int maxSize) {
            array = new int[maxSize];

        }

        public void push(int x) {
            if (index == array.length) {
                return;
            }
            array[index++] = x;
        }

        public int pop() {
            if (index == 0) {
                return -1;
            }
            return array[--index];
        }

        public void increment(int k, int val) {
            int min = Math.min(index, k);
            for (int i = 0; i < min; ++i) {
                array[i] += val;
            }
        }
    }

    // 1381. 设计一个支持增量操作的栈
    class CustomStack2 {
        private int[] stack;
        private int[] increasement;
        private int index;

        public CustomStack2(int maxSize) {
            stack = new int[maxSize];
            increasement = new int[maxSize];
            index = -1;

        }

        public void push(int x) {
            if (index + 1 == stack.length) {
                return;
            }
            stack[++index] = x;

        }

        public int pop() {
            if (index == -1) {
                return -1;
            }
            int res = stack[index] + increasement[index];
            if (index > 0) {
                increasement[index - 1] += increasement[index];
            }
            increasement[index] = 0;
            --index;
            return res;
        }

        public void increment(int k, int val) {
            int min = Math.min(k, index + 1);
            if (min > 0) {
                increasement[min - 1] += val;
            }

        }
    }

    // 1047. 删除字符串中的所有相邻重复项
    public String removeDuplicates1047(String s) {
        StringBuilder res = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (res.length() != 0 && res.charAt(res.length() - 1) == c) {
                res.deleteCharAt(res.length() - 1);
            } else {
                res.append(c);
            }
        }
        return res.toString();

    }

    // 1021. 删除最外层的括号
    public String removeOuterParentheses(String s) {
        Set<Integer> set = new HashSet<>();
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                stack.push(i);
            } else {
                if (stack.size() == 1) {
                    set.add(stack.pop());
                    set.add(i);
                } else {
                    stack.pop();
                }
            }
        }
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < s.length(); ++i) {
            if (!set.contains(i)) {
                result.append(s.charAt(i));
            }
        }
        return result.toString();

    }

    // 1021. 删除最外层的括号
    public String removeOuterParentheses2(String s) {
        StringBuilder result = new StringBuilder();
        int left = 0;
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == '(') {
                if (left++ == 0) {
                    set.add(i);
                }
            } else {
                if (left-- == 1) {
                    set.add(i);
                }
            }
        }
        for (int i = 0; i < s.length(); ++i) {
            if (!set.contains(i)) {
                result.append(s.charAt(i));
            }
        }
        return result.toString();

    }

    // 1021. 删除最外层的括号
    public String removeOuterParentheses3(String s) {
        StringBuilder builder = new StringBuilder();
        int count = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                if (count++ == 0) {
                    continue;
                }
            } else {
                if (--count == 0) {
                    continue;
                }
            }
            builder.append(c);
        }
        return builder.toString();

    }

    // 1021. 删除最外层的括号
    public String removeOuterParentheses4(String s) {
        StringBuilder result = new StringBuilder();
        int level = 0;
        for (char c : s.toCharArray()) {
            if (c == ')') {
                --level;
            }
            if (level >= 1) {
                result.append(c);
            }
            if (c == '(') {
                ++level;
            }

        }
        return result.toString();

    }

    // 496. 下一个更大元素 I
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        int[] dp = new int[10001];
        int[] res = new int[nums1.length];
        Arrays.fill(res, -1);
        Arrays.fill(dp, -1);
        for (int i = 0; i < nums1.length; ++i) {
            dp[nums1[i]] = i;
        }
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < nums2.length; ++i) {
            while (!stack.isEmpty() && nums2[i] > stack.peek()) {
                int num = stack.pop();
                if (dp[num] != -1) {
                    res[dp[num]] = nums2[i];
                }
            }
            stack.push(nums2[i]);
        }
        return res;

    }

    // 面试题 03.02. 栈的最小值 // 剑指 Offer 30. 包含min函数的栈 // 155. 最小栈
    class MinStack {
        List<Integer> stack;
        List<Integer> min;

        /** initialize your data structure here. */
        public MinStack() {
            stack = new ArrayList<>();
            min = new ArrayList<>();

        }

        public void push(int x) {
            min.add(min.isEmpty() ? x : Math.min(min.get(min.size() - 1), x));
            stack.add(x);
        }

        public void pop() {
            min.remove(min.size() - 1);
            stack.remove(stack.size() - 1);
        }

        public int top() {
            return stack.get(stack.size() - 1);
        }

        public int min() {
            return min.get(min.size() - 1);
        }
    }

    // 503. 下一个更大元素 II
    public int[] nextGreaterElements(int[] nums) {
        int[] res = new int[nums.length];
        Arrays.fill(res, -1);
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < nums.length * 2 - 1; ++i) {
            while (!stack.isEmpty() && nums[i % nums.length] > nums[stack.peek()]) {
                res[stack.pop()] = nums[i % nums.length];
            }
            stack.push(i % nums.length);
        }
        return res;

    }

    // 1003. 检查替换后的词是否有效
    public boolean isValid1003(String s) {
        String pattern = "abc";
        StringBuilder builder = new StringBuilder(s);
        while (builder.length() != 0) {
            int index = builder.indexOf(pattern);
            if (index == -1) {
                return false;
            }
            builder.delete(index, index + 3);
        }
        return true;

    }

    // 1003. 检查替换后的词是否有效
    // s = "abcabcababcc"
    public boolean isValid1003_2(String s) {
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == 'c') {
                if (res.length() < 2 || res.charAt(res.length() - 1) != 'b' || res.charAt(res.length() - 2) != 'a') {
                    return false;
                }
                res.setLength(res.length() - 2);
            } else {
                res.append(s.charAt(i));
            }
        }
        return res.length() == 0;

    }

    // 456. 132 模式
    public boolean find132pattern(int[] nums) {
        Stack<Integer> stack = new Stack<>();
        int min = Integer.MIN_VALUE;
        for (int i = nums.length - 1; i >= 0; --i) {
            if (nums[i] < min) {
                return true;
            }
            while (!stack.isEmpty() && nums[i] > stack.peek()) {
                min = stack.pop();
            }
            stack.push(nums[i]);
        }
        return false;

    }

    // 1598. 文件夹操作日志搜集器
    public int minOperations(String[] logs) {
        Stack<String> stack = new Stack<>();
        for (String log : logs) {
            if (log.equals("../")) {
                if (!stack.isEmpty()) {
                    stack.pop();
                }
            } else {
                if (!log.equals("./")) {
                    stack.push(log);
                }
            }
        }
        return stack.size();
    }

    // 1598. 文件夹操作日志搜集器
    public int minOperations_2(String[] logs) {
        int count = 0;
        for (String log : logs) {
            if (log.equals("../")) {
                if (count != 0) {
                    --count;
                }
            } else if (!log.equals("./")) {
                ++count;
            }
        }
        return count;
    }

    // 150. 逆波兰表达式求值
    public int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for (String token : tokens) {
            if ("+".equals(token)) {
                int last = stack.pop();
                int secondToLast = stack.pop();
                stack.push(secondToLast + last);
            } else if ("-".equals(token)) {
                int last = stack.pop();
                int secondToLast = stack.pop();
                stack.push(secondToLast - last);

            } else if ("*".equals(token)) {
                int last = stack.pop();
                int secondToLast = stack.pop();
                stack.push(secondToLast * last);

            } else if ("/".equals(token)) {
                int last = stack.pop();
                int secondToLast = stack.pop();
                stack.push(secondToLast / last);
            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }

    // 150. 逆波兰表达式求值
    public int evalRPN_2(String[] tokens) {
        int index = -1;
        int[] stack = new int[(tokens.length + 1) / 2];
        for (String token : tokens) {
            switch (token) {
                case "+":
                    --index;
                    stack[index] += stack[index + 1];
                    break;
                case "-":
                    --index;
                    stack[index] -= stack[index + 1];
                    break;
                case "*":
                    --index;
                    stack[index] *= stack[index + 1];
                    break;
                case "/":
                    --index;
                    stack[index] /= stack[index + 1];
                    break;
                default:
                    stack[++index] = Integer.parseInt(token);
                    break;
            }
        }
        return stack[index];

    }

    // 735. 行星碰撞
    public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> stack = new Stack<>();
        for (int asteroid : asteroids) {
            if (asteroid > 0) {
                stack.push(asteroid);
            } else {
                if (stack.isEmpty()) {
                    stack.push(asteroid);
                } else {
                    int flag = 0;
                    while (!stack.isEmpty() && stack.peek() > 0) {
                        if (asteroid + stack.peek() < 0) {
                            stack.pop();
                            flag = 0;
                        } else if (asteroid + stack.peek() == 0) {
                            stack.pop();
                            flag = 1;
                            break;
                        } else if (asteroid + stack.peek() > 0) {
                            flag = 1;
                            break;
                        }
                    }
                    if (flag == 0) {
                        stack.push(asteroid);
                    }
                }
            }
        }

        int[] res = new int[stack.size()];
        for (int i = 0; i < stack.size(); ++i) {
            res[i] = stack.get(i);
        }
        return res;

    }

    // 735. 行星碰撞
    public int[] asteroidCollision2(int[] asteroids) {
        Stack<Integer> stack = new Stack<>();
        for (int asteroid : asteroids) {
            collisions: {
                while (!stack.isEmpty() && asteroid < 0 && stack.peek() > 0) {
                    if (stack.peek() + asteroid < 0) {
                        stack.pop();
                        continue;
                    } else if (stack.peek() + asteroid == 0) {
                        stack.pop();
                    }
                    break collisions;
                }
                stack.push(asteroid);
            }
        }
        int[] res = new int[stack.size()];
        for (int i = 0; i < stack.size(); ++i) {
            res[i] = stack.get(i);
        }
        return res;

    }

    // 402. 移掉K位数字
    // 特殊情况：num = 1200 ，k = 3
    public String removeKdigits(String num, int k) {
        if (num.length() == k) {
            return "0";
        }
        Stack<Character> stack = new Stack<>();
        for (int i = 0; i < num.length(); ++i) {
            char c = num.charAt(i);
            while (!stack.isEmpty() && k > 0 && stack.peek() > c) {
                stack.pop();
                --k;
            }
            if (stack.isEmpty() && c == '0') {
                continue;
            }
            stack.push(c);
        }
        StringBuilder res = new StringBuilder();
        if (stack.size() <= k) {
            return "0";
        }
        for (int i = 0; i < stack.size(); ++i) {
            res.append(stack.get(i));
        }
        return res.substring(0, stack.size() - k).toString();

    }

    // 402. 移掉K位数字
    public String removeKdigits2(String num, int k) {
        if (num.length() == k) {
            return "0";
        }
        StringBuilder res = new StringBuilder();
        for (char c : num.toCharArray()) {
            while (res.length() != 0 && c < res.charAt(res.length() - 1) && k > 0) {
                res.setLength(res.length() - 1);
                --k;
            }
            if (res.length() == 0 && c == '0') {
                continue;
            }
            res.append(c);
        }
        // num = 10000, k=1
        if (res.length() <= k) {
            return "0";
        }
        res.setLength(res.length() - k);
        return res.toString();

    }

    // 316. 去除重复字母 // 1081. 不同字符的最小子序列
    public String removeDuplicateLetters(String s) {
        int[] counts = new int[26];
        boolean[] seen = new boolean[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        StringBuilder res = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (!seen[c - 'a']) {
                while (res.length() != 0 && c < res.charAt(res.length() - 1)
                        && counts[res.charAt(res.length() - 1) - 'a'] > 1) {
                    --counts[res.charAt(res.length() - 1) - 'a'];
                    seen[res.charAt(res.length() - 1) - 'a'] = false;
                    res.setLength(res.length() - 1);
                }
                res.append(c);
                seen[c - 'a'] = true;

            } else {
                --counts[c - 'a'];
            }
        }
        return res.toString();

    }

    // 321. 拼接最大数
    public int[] maxNumber(int[] nums1, int[] nums2, int k) {
        int[] res = new int[k];
        String currentMax = "";
        for (int i = 0; i <= k; ++i) {
            int min1 = Math.min(i, nums1.length);
            int min2 = Math.min(k - i, nums2.length);
            if (min1 + min2 < k) {
                continue;
            }
            String s1 = getMaxNum(nums1, i);
            String s2 = getMaxNum(nums2, k - i);
            String current = getMaxMerged(s1, s2);
            if (currentMax.compareTo(current) < 0) {
                currentMax = current;
            }
        }
        for (int i = 0; i < currentMax.length(); ++i) {
            res[i] = currentMax.charAt(i) - '0';
        }
        return res;

    }

    private String getMaxMerged(String s1, String s2) {
        if (s1.length() == 0) {
            return s2;
        } else if (s2.length() == 0) {
            return s1;
        }
        StringBuilder res = new StringBuilder();
        int len = s1.length() + s2.length();
        int index1 = 0;
        int index2 = 0;
        for (int i = 0; i < len; ++i) {
            if (compare(s1, index1, s2, index2) > 0) {
                res.append(s1.charAt(index1++));
            } else {
                res.append(s2.charAt(index2++));
            }
        }
        return res.toString();
    }

    private int compare(String s1, int index1, String s2, int index2) {
        while (index1 < s1.length() && index2 < s2.length()) {
            int diff = s1.charAt(index1) - s2.charAt(index2);
            if (diff != 0) {
                return diff;
            }
            ++index1;
            ++index2;
        }
        return (s1.length() - index1) - (s2.length() - index2);
    }

    private String getMaxNum(int[] nums, int k) {
        if (k == 0) {
            return "";
        }
        int deleteK = nums.length - k;
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < nums.length; ++i) {
            while (res.length() != 0 && nums[i] > (res.charAt(res.length() - 1) - '0') && deleteK > 0) {
                --deleteK;
                res.setLength(res.length() - 1);
            }
            res.append(nums[i]);
        }

        return res.substring(0, res.length() - deleteK).toString();
    }

    // 剑指 Offer 09. 用两个栈实现队列
    class CQueue {
        private Stack<Integer> stack1;
        private Stack<Integer> stack2;

        public CQueue() {
            stack1 = new Stack<>();
            stack2 = new Stack<>();

        }

        public void appendTail(int value) {
            stack1.push(value);
        }

        public int deleteHead() {
            if (stack2.isEmpty()) {
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.pop());
                }
            }
            if (stack2.isEmpty()) {
                return -1;
            }
            return stack2.pop();

        }
    }

    // 面试题 03.04. 化栈为队 // 232. 用栈实现队列
    class MyQueue {
        private Stack<Integer> stack1;
        private Stack<Integer> stack2;

        /** Initialize your data structure here. */
        public MyQueue() {
            stack1 = new Stack<>();
            stack2 = new Stack<>();

        }

        /** Push element x to the back of queue. */
        public void push(int x) {
            stack1.push(x);
        }

        /** Removes the element from in front of queue and returns that element. */
        public int pop() {
            if (stack2.isEmpty()) {
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.pop());
                }
            }
            return stack2.pop();
        }

        /** Get the front element. */
        public int peek() {
            if (stack2.isEmpty()) {
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.pop());
                }
            }
            return stack2.peek();

        }

        /** Returns whether the queue is empty. */
        public boolean empty() {
            return stack1.isEmpty() && stack2.isEmpty();
        }
    }

    // 136. 只出现一次的数字
    public int singleNumber1366(int[] nums) {
        int ans = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            ans ^= nums[i];
        }
        return ans;

    }

    // 225. 用队列实现栈 -- 两个队列
    class MyStack {
        Queue<Integer> queue1;
        Queue<Integer> queue2;

        /** Initialize your data structure here. */
        public MyStack() {
            queue1 = new LinkedList<>();
            queue2 = new LinkedList<>();

        }

        /** Push element x onto stack. */
        public void push(int x) {
            queue2.offer(x);
            while (!queue1.isEmpty()) {
                queue2.offer(queue1.poll());
            }
            Queue<Integer> temp = queue1;
            queue1 = queue2;
            queue2 = temp;
        }

        /** Removes the element on top of the stack and returns that element. */
        public int pop() {
            return queue1.poll();

        }

        /** Get the top element. */
        public int top() {
            return queue1.peek();

        }

        /** Returns whether the stack is empty. */
        public boolean empty() {
            return queue1.isEmpty();
        }
    }

    // 225. 用队列实现栈 -- 一个队列
    class MyStack2 {
        private Queue<Integer> queue;

        /** Initialize your data structure here. */
        public MyStack2() {
            queue = new LinkedList<>();
        }

        /** Push element x onto stack. */
        public void push(int x) {
            int size = queue.size();
            queue.offer(x);
            while (size-- > 0) {
                queue.offer(queue.poll());
            }

        }

        /** Removes the element on top of the stack and returns that element. */
        public int pop() {
            return queue.poll();

        }

        /** Get the top element. */
        public int top() {
            return queue.peek();

        }

        /** Returns whether the stack is empty. */
        public boolean empty() {
            return queue.isEmpty();
        }
    }

    // 895. 最大频率栈
    class FreqStack {
        private Map<Integer, Integer> count;
        private Map<Integer, Stack<Integer>> freq;
        private int maxFreq;

        public FreqStack() {
            count = new HashMap<>();
            freq = new HashMap<>();

        }

        public void push(int val) {
            count.put(val, count.getOrDefault(val, 0) + 1);
            int num = count.get(val);
            freq.computeIfAbsent(num, k -> new Stack<>()).push(val);
            maxFreq = Math.max(maxFreq, num);
        }

        public int pop() {
            Stack<Integer> maxFreqStack = freq.get(maxFreq);
            int res = maxFreqStack.pop();
            count.put(res, count.get(res) - 1);
            if (maxFreqStack.isEmpty()) {
                --maxFreq;
            }
            return res;
        }
    }

    // 1886. 判断矩阵经轮转后是否一致
    public boolean findRotation(int[][] mat, int[][] target) {
        for (int i = 0; i < 4; ++i) {
            rotate90Degree(mat);
            if (matEqualsTarget(mat, target)) {
                return true;
            }
        }
        return false;

    }

    private boolean matEqualsTarget(int[][] mat, int[][] target) {
        int n = mat.length;
        for (int i = 0; i < n; ++i) {
            if (!Arrays.equals(mat[i], target[i])) {
                return false;
            }
        }
        return true;
    }

    private void rotate90Degree(int[][] mat) {
        int n = mat.length;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                int temp = mat[i][j];
                mat[i][j] = mat[j][i];
                mat[j][i] = temp;
            }
        }
        for (int i = 0; i < n; ++i) {
            reverse1886(mat[i]);
        }
    }

    private void reverse1886(int[] nums) {
        int i = 0;
        int j = nums.length - 1;
        while (i < j) {
            int temp = nums[i];
            nums[i] = nums[j];
            nums[j] = temp;
            ++i;
            --j;
        }
    }

    // 1888. 使二进制字符串字符交替的最少反转次数
    // public int minFlips(String s) {
    // int count = Integer.MAX_VALUE;
    // for (int i = 0; i < s.length(); ++i) {
    // count = Math.min(count, getMinCount(ss.substring(i, i + s.length())));
    // }
    // return count;

    // }

    // private int getMinCount(String string) {
    // int count1 = 0;
    // int count2 = 0;
    // for (int i = 0; i < string.length(); ++i) {
    // if (i % 2 == (string.charAt(i) - '0')) {
    // ++count1;
    // } else {
    // ++count2;
    // }
    // }
    // return Math.min(count1, count2);

    // }

    class MyQueue232 {
        Stack<Integer> stack1;
        Stack<Integer> stack2;

        /** Initialize your data structure here. */
        public MyQueue232() {
            stack1 = new Stack<>();
            stack2 = new Stack<>();

        }

        /** Push element x to the back of queue. */
        public void push(int x) {
            stack1.push(x);

        }

        /** Removes the element from in front of queue and returns that element. */
        public int pop() {
            if (stack2.isEmpty()) {
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.pop());
                }
            }
            return stack2.pop();

        }

        /** Get the front element. */
        public int peek() {
            if (stack2.isEmpty()) {
                while (!stack1.isEmpty()) {
                    stack2.push(stack1.pop());
                }
            }
            return stack2.peek();
        }

        /** Returns whether the queue is empty. */
        public boolean empty() {
            return stack1.isEmpty() && stack2.isEmpty();
        }
    }

    // 223. 矩形面积
    public int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int area1 = (C - A) * (D - B);
        int area2 = (G - E) * (H - F);
        int maxLeft = Math.max(A, E);
        int minRight = Math.min(C, G);
        int minTop = Math.min(D, H);
        int maxBottom = Math.max(B, F);
        if (maxLeft >= minRight || minTop <= maxBottom) {
            return area1 + area2;
        }
        return area1 + area2 - (minRight - maxLeft) * (minTop - maxBottom);
    }

    // 剑指 Offer 59 - II. 队列的最大值
    class MaxQueue {
        Queue<Integer> queue;
        Deque<Integer> deque;

        public MaxQueue() {
            queue = new LinkedList<>();
            deque = new LinkedList<>();

        }

        public int max_value() {
            if (queue.isEmpty()) {
                return -1;
            }
            return deque.peekFirst();

        }

        public void push_back(int value) {
            while (!deque.isEmpty() && deque.peekLast() < value) {
                deque.pollLast();
            }
            deque.offerLast(value);
            queue.offer(value);
        }

        public int pop_front() {
            if (queue.isEmpty()) {
                return -1;
            }
            int ans = queue.poll();
            if (ans == deque.peekFirst()) {
                deque.pollFirst();
            }
            return ans;

        }
    }

    // 299. 猜数字游戏（与“面试题 16.15. 珠玑妙算” 类似）
    public String getHint(String secret, String guess) {
        int[] counts = new int[10];
        char[] arraySecret = secret.toCharArray();
        char[] arrayGuess = guess.toCharArray();
        int bulls = 0;
        int cows = 0;
        for (int i = 0; i < arraySecret.length; ++i) {
            if (arraySecret[i] == arrayGuess[i]) {
                arraySecret[i] = 'x';
                arrayGuess[i] = 'x';
                ++bulls;
            } else {
                ++counts[arraySecret[i] - '0'];
            }
        }
        for (int i = 0; i < arrayGuess.length; ++i) {
            if (Character.isDigit(arrayGuess[i]) && counts[arrayGuess[i] - '0'] > 0) {
                --counts[arrayGuess[i] - '0'];
                ++cows;
            }
        }
        return bulls + "A" + cows + "B";

    }

    // 299. 猜数字游戏（与“面试题 16.15. 珠玑妙算” 类似）
    public String getHint2(String secret, String guess) {
        int[] counts = new int[10];
        int bulls = 0;
        int bullsAndCows = 0;
        for (int i = 0; i < secret.length(); ++i) {
            if (secret.charAt(i) == guess.charAt(i)) {
                ++bulls;
            }
            ++counts[secret.charAt(i) - '0'];
        }
        for (int i = 0; i < guess.length(); ++i) {
            if (counts[guess.charAt(i) - '0'] > 0) {
                --counts[guess.charAt(i) - '0'];
                ++bullsAndCows;
            }
        }
        return bulls + "A" + (bullsAndCows - bulls) + "B";

    }

    // 389. 找不同
    public char findTheDifference(String s, String t) {
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        for (char c : t.toCharArray()) {
            if (--counts[c - 'a'] < 0) {
                return c;
            }
        }
        return 'a';

    }

    // 389. 找不同
    public char findTheDifference2(String s, String t) {
        int res = 0;
        for (char c : (s + t).toCharArray()) {
            res ^= c;
        }
        return (char) res;

    }

    // 386. 字典序排数--未超时，但不是最好解法
    public List<Integer> lexicalOrder(int n) {
        int i = 1;
        List<Integer> list = new ArrayList<>();
        while (i <= n) {
            list.add(i++);
        }
        Collections.sort(list, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                String s1 = String.valueOf(o1);
                String s2 = String.valueOf(o2);
                return s1.compareTo(s2);
            }
        });
        return list;

    }

    // 171. Excel表列序号
    public int titleToNumber(String columnTitle) {
        int bit = 0;
        final int SCALE = 26;
        int res = 0;
        for (int i = columnTitle.length() - 1; i >= 0; --i) {
            int num = columnTitle.charAt(i) - 'A' + 1;
            res += num * Math.pow(SCALE, bit++);
        }
        return res;

    }

    // 868. 二进制间距
    public int binaryGap(int n) {
        int max = 0;
        int pos = -1;
        while (n > 0) {
            int curPos = (int) (Math.log(n) / Math.log(2));
            if (pos != -1) {
                max = Math.max(max, pos - curPos);
            }
            pos = curPos;
            n -= Math.pow(2, curPos);
        }
        return max;

    }

    // 868. 二进制间距
    public int binaryGap2(int n) {
        int pos = -1;
        int res = 0;
        int times = 0;
        while (n != 0) {
            if (n % 2 == 1) {
                if (pos != -1) {
                    res = Math.max(times - pos, res);
                }
                pos = times;
            }
            ++times;
            n /= 2;
        }
        return res;

    }

    // 5768. 找到需要补充粉笔的学生编号
    public int chalkReplacer(int[] chalk, int k) {
        long sum = 0;
        for (int cha : chalk) {
            sum += cha;
        }
        long remain = k % sum;
        for (int i = 0; i < chalk.length; ++i) {
            if (remain < chalk[i]) {
                return i;
            }
            remain -= chalk[i];
        }
        return -1;

    }

    // This is the interface that allows for creating nested lists. // You should
    // not implement it, or speculate about its implementation
    public class NestedInteger { // Constructor initializes an empty nested list.
        public NestedInteger() {

        }

        public NestedInteger(int val) {

        }

        // Constructor initializes a single integer. public NestedInteger(int value);

        // @return true if this NestedInteger holds a single integer, rather than a
        // nested list.
        public boolean isInteger() {
            return false;
        }

        // @return the single integer that this NestedInteger holds, if it holds a
        // single integer // Return null if this NestedInteger holds a nested list
        public Integer getInteger() {
            return null;
        }

        // Set this NestedInteger to hold a single integer.
        public void setInteger(int value) {
        }

        // Set this NestedInteger to hold a nested list and adds a nested integer to
        // it.
        public void add(NestedInteger ni) {
        }

        // @return the nested list that this NestedInteger holds, if it holds a
        // nested list
        // Return null if this NestedInteger holds a single integer
        public List<NestedInteger> getList() {
            return null;

        }
    }

    // 385. 迷你语法分析器
    public NestedInteger deserialize(String s) {

        if (s.charAt(0) != '[') {
            return new NestedInteger(Integer.parseInt(s));
        }
        int i = 0;
        Stack<NestedInteger> stack = new Stack<>();
        while (i < s.length()) {
            if (s.charAt(i) == '[') {
                stack.push(new NestedInteger());
            } else if (Character.isDigit(s.charAt(i)) || s.charAt(i) == '-') {
                int sign = 1;
                if (s.charAt(i) == '-') {
                    sign = -1;
                    ++i;
                }
                int num = 0;
                while (Character.isDigit(s.charAt(i))) {
                    num = num * 10 + (s.charAt(i) - '0');
                    ++i;
                }
                --i;
                stack.peek().add(new NestedInteger(sign * num));
            } else if (s.charAt(i) == ']') {
                if (stack.size() > 1) {
                    NestedInteger nestedInteger = stack.pop();
                    stack.peek().add(nestedInteger);
                }
            }
            ++i;

        }
        return stack.pop();

    }

    // 591. 标签验证器
    // outside变量表示最外层是否有且仅有一组标签 如“"<A></A><B></B>"”最外层有两组标签，则不合法
    // 或者outside变量可以理解为标签外是否包含了不合法的内容 如 <A></A>CCC> 其中“CCC>”是不合法的
    private boolean outside;
    private Stack<String> stack = new Stack<>();

    public boolean isValid591(String code) {
        if (code.charAt(0) != '<' || code.charAt(code.length() - 1) != '>') {
            return false;
        }
        for (int i = 0; i < code.length(); ++i) {
            int closeIndex = 0;
            boolean ending = false;
            if (stack.isEmpty() && outside) {
                return false;
            }
            if (code.charAt(i) == '<') {
                if (!stack.isEmpty() && code.charAt(i + 1) == '!') {
                    closeIndex = code.indexOf("]]>", i + 2);
                    if (closeIndex < 0 || !isValidCDATA(code.substring(i + 2, closeIndex))) {
                        return false;
                    }
                } else {
                    if (code.charAt(i + 1) == '/') {
                        ending = true;
                        ++i;
                    }
                    closeIndex = code.indexOf(">", i + 1);
                    if (closeIndex < 0 || !isValidTagName(code.substring(i + 1, closeIndex), ending)) {
                        return false;
                    }
                }
                i = closeIndex;
            }
        }
        return stack.isEmpty() && outside;

    }

    private boolean isValidCDATA(String string) {
        return string.indexOf("[CDATA[") == 0;
    }

    private boolean isValidTagName(String tagName, boolean ending) {
        if (tagName.length() < 1 || tagName.length() > 9) {
            return false;
        }
        for (int i = 0; i < tagName.length(); ++i) {
            if (!Character.isUpperCase(tagName.charAt(i))) {
                return false;
            }
        }
        if (ending) {
            if (!stack.isEmpty() && stack.peek().equals(tagName)) {
                stack.pop();
            } else {
                return false;
            }
        } else {
            outside = true;
            stack.push(tagName);
        }

        return true;
    }

    // 636. 函数的独占时间
    public int[] exclusiveTime(int n, List<String> logs) {
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[n];
        String[] log0 = logs.get(0).split(":");
        int prev = Integer.parseInt(log0[2]);
        stack.push(Integer.parseInt(log0[0]));
        for (int i = 1; i < logs.size(); ++i) {
            String[] log = logs.get(i).split(":");
            if (log[1].equals("start")) {
                if (!stack.isEmpty()) {
                    res[stack.peek()] += Integer.parseInt(log[2]) - prev;
                }
                prev = Integer.parseInt(log[2]);
                stack.push(Integer.parseInt(log[0]));
            } else {
                res[stack.pop()] += Integer.parseInt(log[2]) - prev + 1;
                prev = Integer.parseInt(log[2]) + 1;
            }
        }
        return res;

    }

    // 1673. 找出最具竞争力的子序列
    public int[] mostCompetitive(int[] nums, int k) {
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[k];
        for (int i = 0; i < nums.length; ++i) {
            while (!stack.isEmpty() && nums[i] < stack.peek() && (stack.size() + nums.length - i - 1) >= k) {
                stack.pop();
            }
            stack.push(nums[i]);
        }
        for (int i = 0; i < k; ++i) {
            res[i] = stack.get(i);
        }
        return res;

    }

    // 1410. HTML 实体解析器
    private int i;

    public String entityParser(String text) {
        StringBuilder res = new StringBuilder();
        while (i < text.length()) {
            if (text.charAt(i) == '&') {
                char parse = parse(text.substring(i, text.length()));
                if (parse != ' ') {
                    res.append(parse);
                } else {
                    res.append(text.charAt(i));
                }
            } else {
                res.append(text.charAt(i));
            }
            ++i;
        }
        return res.toString();

    }

    // 双引号：字符实体为 &quot; ，对应的字符是 " 。
    // 单引号：字符实体为 &apos; ，对应的字符是 ' 。
    // 与符号：字符实体为 &amp; ，对应对的字符是 & 。
    // 大于号：字符实体为 &gt; ，对应的字符是 > 。
    // 小于号：字符实体为 &lt; ，对应的字符是 < 。
    // 斜线号：字符实体为 &frasl; ，对应的字符是 / 
    private char parse(String string) {
        if (string.indexOf("&quot;") == 0) {
            i += 5;
            return '\"';
        }
        if (string.indexOf("&apos;") == 0) {
            i += 5;
            return '\'';
        }
        if (string.indexOf("&amp;") == 0) {
            i += 4;
            return '&';
        }
        if (string.indexOf("&gt;") == 0) {
            i += 3;
            return '>';
        }
        if (string.indexOf("&lt;") == 0) {
            i += 3;
            return '<';
        }
        if (string.indexOf("&frasl;") == 0) {
            i += 6;
            return '/';
        }

        return ' ';
    }

    // 1410. HTML 实体解析器
    // 双引号：字符实体为 &quot; ，对应的字符是 " 。
    // 单引号：字符实体为 &apos; ，对应的字符是 ' 。
    // 与符号：字符实体为 &amp; ，对应对的字符是 & 。
    // 大于号：字符实体为 &gt; ，对应的字符是 > 。
    // 小于号：字符实体为 &lt; ，对应的字符是 < 。
    // 斜线号：字符实体为 &frasl; ，对应的字符是 / 
    public String entityParser2(String text) {
        return text.replace("&quot;", "\"").replace("&apos;", "\'").replace("&gt;", ">").replace("&lt;", "<")
                .replace("&frasl;", "/").replace("&amp;", "&");
    }

    // 482. 密钥格式化
    public String licenseKeyFormatting(String s, int k) {
        s = s.replace("-", "").toUpperCase();
        int cur = 0;
        StringBuilder res = new StringBuilder();
        for (int i = s.length() - 1; i >= 0; --i) {
            res.append(s.charAt(i));
            ++cur;
            if (cur % k == 0 && i != 0) {
                res.append("-");
            }
        }
        return res.reverse().toString();

    }

    // 599. 两个列表的最小索引总和
    public String[] findRestaurant(String[] list1, String[] list2) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < list1.length; ++i) {
            map.put(list1[i], i);
        }
        List<String> res = new ArrayList<>();
        int minSumIndex = Integer.MAX_VALUE;
        int curSumIndex = 0;
        for (int i = 0; i < list2.length; ++i) {
            if (map.containsKey(list2[i])) {
                curSumIndex = map.get(list2[i]) + i;
                if (curSumIndex < minSumIndex) {
                    res.clear();
                    res.add(list2[i]);
                    minSumIndex = curSumIndex;
                } else if (curSumIndex == minSumIndex) {
                    res.add(list2[i]);
                }
            }
        }
        return res.toArray(new String[res.size()]);

    }

    // 593. 有效的正方形
    public boolean validSquare(int[] p1, int[] p2, int[] p3, int[] p4) {
        int[][] points = { p1, p2, p3, p4 };
        Arrays.sort(points, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[0] == o2[0]) {
                    return o1[1] - o2[1];
                }
                return o1[0] - o2[0];
            }
        });
        return calculateDistance(points[0], points[1]) != 0
                && calculateDistance(points[0], points[1]) == calculateDistance(points[0], points[2])
                && calculateDistance(points[0], points[2]) == calculateDistance(points[2], points[3])
                && calculateDistance(points[2], points[3]) == calculateDistance(points[1], points[3])
                && calculateDistance(points[1], points[2]) == calculateDistance(points[0], points[3]);

    }

    private int calculateDistance(int[] point1, int[] point2) {
        return (point2[1] - point1[1]) * (point2[1] - point1[1]) + (point2[0] - point1[0]) * (point2[0] - point1[0]);
    }

    // 594. 最长和谐子序列
    public int findLHS(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        int res = 0;
        for (int key : map.keySet()) {
            if (map.containsKey(key + 1)) {
                res = Math.max(res, map.get(key) + map.get(key + 1));
            }
        }
        return res;

    }

    // 594. 最长和谐子序列
    public int findLHS2(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
            if (map.containsKey(num + 1)) {
                res = Math.max(res, map.get(num) + map.get(num + 1));
            }
            if (map.containsKey(num - 1)) {
                res = Math.max(res, map.get(num) + map.get(num - 1));
            }
        }
        return res;

    }

    // 1905. 统计子岛屿
    public int countSubIslands(int[][] grid1, int[][] grid2) {
        int m = grid2.length;
        int n = grid2[0].length;
        Union1905 union = new Union1905(n * m);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid2[i][j] == 1) {
                    if (i + 1 < m && grid2[i + 1][j] == 1) {
                        union.union(getIndex1905(n, i, j), getIndex1905(n, i + 1, j));
                    }
                    if (j + 1 < n && grid2[i][j + 1] == 1) {
                        union.union(getIndex1905(n, i, j), getIndex1905(n, i, j + 1));
                    }
                }
            }
        }
        Set<Integer> not = new HashSet<>();
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid2[i][j] == 1) {
                    int root = union.getRoot(getIndex1905(n, i, j));
                    if (!not.contains(root)) {
                        set.add(root);
                    }
                    if (grid1[i][j] == 0) {
                        not.add(root);
                        set.remove(root);
                    }
                }
            }
        }
        return set.size();

    }

    private int getIndex1905(int n, int i, int j) {
        return i * n + j;
    }

    public class Union1905 {
        private int[] parent;
        private int[] rank;

        public Union1905(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            rank = new int[n];
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
        }

    }

    // 1906. 查询差绝对值的最小值
    public int[] minDifference(int[] nums, int[][] queries) {
        int[][] dp = new int[nums.length][101];
        int[] arr = new int[101];
        for (int i = 0; i < nums.length; ++i) {
            ++arr[nums[i]];
            dp[i] = arr.clone();
        }
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            int left = queries[i][0];
            int right = queries[i][1];
            int[] query = new int[101];
            if (left == 0) {
                query = dp[right];
            } else {
                int[] queryRight = dp[right];
                int[] queryLeft = dp[left - 1];
                for (int j = 0; j < query.length; ++j) {
                    query[j] = queryRight[j] - queryLeft[j];
                }
            }
            int min = Integer.MAX_VALUE;
            int last = -1;
            for (int j = 0; j < query.length; ++j) {
                if (query[j] != 0) {
                    if (last != -1) {
                        min = Math.min(min, j - last);
                    }
                    last = j;
                }
            }
            res[i] = min == Integer.MAX_VALUE ? -1 : min;
        }
        return res;

    }

    // 892. 三维形体的表面积
    public int surfaceArea(int[][] grid) {
        int res = 0;
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                int value = grid[i][j];
                if (value > 0) {
                    res += value * 4 + 2;
                    if (i > 0) {
                        res -= Math.min(value, grid[i - 1][j]) * 2;
                    }
                    if (j > 0) {
                        res -= Math.min(value, grid[i][j - 1]) * 2;
                    }
                }
            }
        }
        return res;

    }

    // 1020. 飞地的数量
    public int numEnclaves(int[][] grid) {
        int res = 0;
        int m = grid.length;
        int n = grid[0].length;
        Union1020 union = new Union1020(n * m + 1);
        int dummy = m * n;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    if (i == 0 || j == 0 || i == m - 1 || j == n - 1) {
                        union.union(getIndex1020(n, i, j), dummy);
                    } else {
                        if (grid[i + 1][j] == 1) {
                            union.union(getIndex1020(n, i, j), getIndex1020(n, i + 1, j));
                        }
                        if (grid[i - 1][j] == 1) {
                            union.union(getIndex1020(n, i, j), getIndex1020(n, i - 1, j));
                        }
                        if (grid[i][j + 1] == 1) {
                            union.union(getIndex1020(n, i, j), getIndex1020(n, i, j + 1));
                        }
                        if (grid[i][j - 1] == 1) {
                            union.union(getIndex1020(n, i, j), getIndex1020(n, i, j - 1));
                        }
                    }
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    if (!union.isConnected(getIndex1020(n, i, j), dummy)) {
                        ++res;
                    }
                }
            }
        }
        return res;

    }

    public class Union1020 {
        private int[] rank;
        private int[] parent;

        public Union1020(int n) {
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

    private int getIndex1020(int n, int i, int j) {
        return i * n + j;
    }

    // 1254. 统计封闭岛屿的数目
    public int closedIsland(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Union1254 union = new Union1254(m * n + 1);
        int dummy = m * n;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                // 0是陆地 1是水域
                if (grid[i][j] == 0) {
                    if (i == 0 || j == 0 || i == m - 1 || j == n - 1) {
                        union.union(getIndex1254(n, i, j), dummy);
                    } else {
                        if (grid[i + 1][j] == 0) {
                            union.union(getIndex1254(n, i, j), getIndex1254(n, i + 1, j));
                        }
                        if (grid[i - 1][j] == 0) {
                            union.union(getIndex1254(n, i, j), getIndex1254(n, i - 1, j));
                        }
                        if (grid[i][j + 1] == 0) {
                            union.union(getIndex1254(n, i, j), getIndex1254(n, i, j + 1));
                        }
                        if (grid[i][j - 1] == 0) {
                            union.union(getIndex1254(n, i, j), getIndex1254(n, i, j - 1));
                        }
                    }
                }
            }
        }
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0 && !union.isConnected(getIndex1254(n, i, j), dummy)) {
                    int root = union.getRoot(getIndex1254(n, i, j));
                    set.add(root);
                }
            }
        }
        return set.size();

    }

    private int getIndex1254(int n, int i, int j) {
        return n * i + j;
    }

    public class Union1254 {
        private int[] rank;
        private int[] parent;

        public Union1254(int n) {
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

    // 1391. 检查网格中是否存在有效路径

    // 1 表示连接左单元格和右单元格的街道。
    // 2 表示连接上单元格和下单元格的街道。
    // 3 表示连接左单元格和下单元格的街道。
    // 4 表示连接右单元格和下单元格的街道。
    // 5 表示连接左单元格和上单元格的街道。
    // 6 表示连接右单元格和上单元格的街道。

    public boolean hasValidPath(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Union1391 union = new Union1391(m * n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                switch (grid[i][j]) {
                    case 1:
                        // 与左边相连
                        mergeLeft(union, grid, i, j);
                        // 与右边相连
                        mergeRight(union, grid, i, j);
                        break;
                    case 2:
                        // 与上边相连
                        mergeUp(union, grid, i, j);
                        // 与下边相连
                        mergeDown(union, grid, i, j);
                        break;
                    case 3:
                        // 与左边相连
                        mergeLeft(union, grid, i, j);
                        // 与下边相连
                        mergeDown(union, grid, i, j);
                        break;
                    case 4:
                        // 与右边相连
                        mergeRight(union, grid, i, j);
                        // 与下边相连
                        mergeDown(union, grid, i, j);
                        break;
                    case 5:
                        // 与左边相连
                        mergeLeft(union, grid, i, j);
                        // 与上边相连
                        mergeUp(union, grid, i, j);
                        break;
                    case 6:
                        // 与上边相连
                        mergeUp(union, grid, i, j);
                        // 与右边相连
                        mergeRight(union, grid, i, j);
                        break;
                }
                if (union.isConnected(getIndex1391(n, 0, 0), getIndex1391(n, m - 1, n - 1))) {
                    return true;
                }
            }
        }
        return false;

    }

    private void mergeLeft(Union1391 union, int[][] grid, int i, int j) {
        int n = grid[0].length;
        if (j - 1 >= 0 && (grid[i][j - 1] == 1 || grid[i][j - 1] == 4 || grid[i][j - 1] == 6)) {
            union.union(getIndex1391(n, i, j), getIndex1391(n, i, j - 1));
        }
    }

    private void mergeRight(Union1391 union, int[][] grid, int i, int j) {
        int n = grid[0].length;
        if (j + 1 < n && (grid[i][j + 1] == 1 || grid[i][j + 1] == 3 || grid[i][j + 1] == 5)) {
            union.union(getIndex1391(n, i, j), getIndex1391(n, i, j + 1));
        }
    }

    private void mergeUp(Union1391 union, int[][] grid, int i, int j) {
        int n = grid[0].length;
        if (i - 1 >= 0 && (grid[i - 1][j] == 2 || grid[i - 1][j] == 3 || grid[i - 1][j] == 4)) {
            union.union(getIndex1391(n, i, j), getIndex1391(n, i - 1, j));
        }
    }

    private void mergeDown(Union1391 union, int[][] grid, int i, int j) {
        int m = grid.length;
        int n = grid[0].length;
        if (i + 1 < m && (grid[i + 1][j] == 2 || grid[i + 1][j] == 5 || grid[i + 1][j] == 6)) {
            union.union(getIndex1391(n, i, j), getIndex1391(n, i + 1, j));
        }
    }

    private int getIndex1391(int n, int i, int j) {
        return i * n + j;
    }

    public class Union1391 {
        private int[] rank;
        private int[] parent;

        public Union1391(int n) {
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

    // 1559. 二维网格图中探测环
    public boolean containsCycle(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Union1559 union = new Union1559(m * n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i > 0 && grid[i - 1][j] == grid[i][j]) {
                    int index1 = getIndex1559(n, i, j);
                    int index2 = getIndex1559(n, i - 1, j);
                    if (!union.isConnected(index1, index2)) {
                        union.union(index1, index2);
                    } else {
                        return true;
                    }
                }
                if (j > 0 && grid[i][j - 1] == grid[i][j]) {
                    int index1 = getIndex1559(n, i, j);
                    int index2 = getIndex1559(n, i, j - 1);
                    if (!union.isConnected(index1, index2)) {
                        union.union(index1, index2);
                    } else {
                        return true;
                    }
                }
            }
        }
        return false;

    }

    private int getIndex1559(int n, int i, int j) {
        return i * n + j;
    }

    public class Union1559 {
        private int[] rank;
        private int[] parent;

        public Union1559(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            rank = new int[n];
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
        }
    }

    // 1584. 连接所有点的最小费用 Kruskal算法--用于获取一张完全图的最小生成树
    public int minCostConnectPoints(int[][] points) {

        int n = points.length;
        List<Point1584> list = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                list.add(new Point1584(getManhattanDistance(points, i, j), i, j));
            }
        }
        Collections.sort(list, new Comparator<Point1584>() {

            @Override
            public int compare(Point1584 o1, Point1584 o2) {
                return o1.getFee() - o2.getFee();
            }
        });
        Union1584 union = new Union1584(n);

        int res = 0;
        for (Point1584 point : list) {
            int fee = point.getFee();
            int pointIndex1 = point.getPointIndex1();
            int pointIndex2 = point.getPointIndex2();
            if (!union.isConnected(pointIndex1, pointIndex2)) {
                union.union(pointIndex1, pointIndex2);
                res += fee;
            }
        }
        return res;

    }

    private int getManhattanDistance(int[][] points, int i, int j) {
        return Math.abs(points[i][0] - points[j][0]) + Math.abs(points[i][1] - points[j][1]);
    }

    public class Point1584 {
        private int fee;
        private int pointIndex1;
        private int pointIndex2;

        public Point1584(int fee, int pointIndex1, int pointIndex2) {
            this.setFee(fee);
            this.setPointIndex1(pointIndex1);
            this.setPointIndex2(pointIndex2);
        }

        public int getPointIndex2() {
            return pointIndex2;
        }

        public void setPointIndex2(int pointIndex2) {
            this.pointIndex2 = pointIndex2;
        }

        public int getPointIndex1() {
            return pointIndex1;
        }

        public void setPointIndex1(int pointIndex1) {
            this.pointIndex1 = pointIndex1;
        }

        public int getFee() {
            return fee;
        }

        public void setFee(int fee) {
            this.fee = fee;
        }

    }

    public class Union1584 {
        private int[] parent;
        private int[] rank;

        public Union1584(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            rank = new int[n];
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
        }

    }

    // 1489. 找到最小生成树里的关键边和伪关键边
    public List<List<Integer>> findCriticalAndPseudoCriticalEdges(int n, int[][] edges) {
        int[][] newEdges = new int[edges.length][4];
        for (int i = 0; i < edges.length; ++i) {
            for (int j = 0; j < edges[i].length; ++j) {
                newEdges[i][j] = edges[i][j];
            }
            newEdges[i][3] = i;
        }
        Arrays.sort(newEdges, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }
        });
        Union1489 unionMin = new Union1489(n);
        int m = edges.length;
        // 计算最小生成树的权值
        int minValue = 0;
        for (int i = 0; i < m; ++i) {
            for (int[] newEdge : newEdges) {
                if (!unionMin.isConnected(newEdge[0], newEdge[1])) {
                    unionMin.union(newEdge[0], newEdge[1]);
                    minValue += newEdge[2];
                }
            }
        }
        List<List<Integer>> res = new ArrayList<>();
        for (int i = 0; i < 2; ++i) {
            res.add(new ArrayList<>());
        }
        for (int i = 0; i < m; ++i) {
            Union1489 unionKey = new Union1489(n);
            int v = 0;
            for (int j = 0; j < m; ++j) {
                // 刨去i边
                if (i != j && !unionKey.isConnected(newEdges[j][0], newEdges[j][1])) {
                    unionKey.union(newEdges[j][0], newEdges[j][1]);
                    v += newEdges[j][2];
                }
            }
            // 刨去i边 当前图不是一棵树 或 是一棵树，但不是最小生成树，则i边是一条关键边
            if (unionKey.getCount() != 1 || v > minValue) {
                res.get(0).add(newEdges[i][3]);
                continue;
            }

            // 至此，刨去i边，是一颗最小生成树
            // 因此，i边不是关键边，但可能是伪关键边 或 非关键边
            unionKey = new Union1489(n);
            v = 0;
            unionKey.union(newEdges[i][0], newEdges[i][1]);
            v += newEdges[i][2];
            for (int j = 0; j < m; ++j) {
                if (i != j && !unionKey.isConnected(newEdges[j][0], newEdges[j][1])) {
                    unionKey.union(newEdges[j][0], newEdges[j][1]);
                    v += newEdges[j][2];
                }
            }
            // 加入i边，可构成最小生成树
            if (v == minValue) {
                res.get(1).add(newEdges[i][3]);
            }
        }
        return res;

    }

    public class Union1489 {
        private int[] parent;
        private int[] rank;
        private int count;

        public Union1489(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            rank = new int[n];
            Arrays.fill(rank, 1);
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

    // 面试题 16.19. 水域大小
    public int[] pondSizes(int[][] land) {
        int m = land.length;
        int n = land[0].length;
        Union16_19 union = new Union16_19(m * n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (land[i][j] == 0) {
                    // 往上合并
                    if (i > 0 && land[i - 1][j] == 0) {
                        union.union(getIndex16_19(n, i, j), getIndex16_19(n, i - 1, j));
                    }
                    // 往左合并
                    if (j > 0 && land[i][j - 1] == 0) {
                        union.union(getIndex16_19(n, i, j), getIndex16_19(n, i, j - 1));
                    }
                    // 往左上合并
                    if (i > 0 && j > 0 && land[i - 1][j - 1] == 0) {
                        union.union(getIndex16_19(n, i, j), getIndex16_19(n, i - 1, j - 1));
                    }
                    // 往右上合并
                    if (i > 0 && j + 1 < n && land[i - 1][j + 1] == 0) {
                        union.union(getIndex16_19(n, i, j), getIndex16_19(n, i - 1, j + 1));
                    }
                }
            }
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (land[i][j] == 0) {
                    int index = getIndex16_19(n, i, j);
                    int root = union.getRoot(index);
                    map.put(root, map.getOrDefault(root, 0) + 1);
                }
            }
        }
        if (map.isEmpty()) {
            return new int[] {};
        }
        int[] res = new int[map.size()];
        int index = 0;
        for (int count : map.values()) {
            res[index++] = count;
        }
        Arrays.sort(res);
        return res;

    }

    private int getIndex16_19(int n, int i, int j) {
        return n * i + j;
    }

    public class Union16_19 {
        private int[] parent;
        private int[] rank;

        public Union16_19(int n) {
            parent = new int[n];
            rank = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
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
        }
    }

    // LCS 03. 主题空间
    public int largestArea(String[] grid) {
        int m = grid.length;
        int n = grid[0].length();
        UnionLCS03 union = new UnionLCS03(m * n + 1);
        int dummy = m * n;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                char c = grid[i].charAt(j);
                if (c == '0' || (i == 0 || j == 0 || i == m - 1 || j == n - 1)) {
                    union.union(getIndexLCS03(n, i, j), dummy);
                } else {
                    if (grid[i].charAt(j - 1) == '0' || c == grid[i].charAt(j - 1)) {
                        union.union(getIndexLCS03(n, i, j), getIndexLCS03(n, i, j - 1));
                    }
                    if (grid[i].charAt(j + 1) == '0' || c == grid[i].charAt(j + 1)) {
                        union.union(getIndexLCS03(n, i, j), getIndexLCS03(n, i, j + 1));
                    }
                    if (grid[i - 1].charAt(j) == '0' || c == grid[i - 1].charAt(j)) {
                        union.union(getIndexLCS03(n, i, j), getIndexLCS03(n, i - 1, j));
                    }
                    if (grid[i + 1].charAt(j) == '0' || c == grid[i + 1].charAt(j)) {
                        union.union(getIndexLCS03(n, i, j), getIndexLCS03(n, i + 1, j));
                    }
                }
            }
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int index = getIndexLCS03(n, i, j);
                int root = union.getRoot(index);
                if (!union.isConnected(root, dummy)) {
                    map.put(root, map.getOrDefault(root, 0) + 1);
                }
            }
        }
        if (map.isEmpty()) {
            return 0;
        }
        return Collections.max(map.values());

    }

    private int getIndexLCS03(int n, int i, int j) {
        return i * n + j;
    }

    public class UnionLCS03 {
        private int[] parent;
        private int[] rank;

        public UnionLCS03(int n) {
            parent = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
            }
            rank = new int[n];
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
        }

    }

    // 1614. 括号的最大嵌套深度
    public int maxDepth(String s) {
        int max = 0;
        int leftCount = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                max = Math.max(max, ++leftCount);
            } else if (c == ')') {
                --leftCount;
            }
        }
        return max;

    }

    // 1653. 使字符串平衡的最少删除次数
    public int minimumDeletions(String s) {
        Stack<Character> stack = new Stack<>();
        int res = 0;
        for (char c : s.toCharArray()) {
            if (c == 'b') {
                stack.push(c);
            } else if (!stack.isEmpty()) {
                ++res;
                stack.pop();
            }
        }
        return res;

    }

    // 1653. 使字符串平衡的最少删除次数
    public int minimumDeletions2(String s) {
        int bCount = 0;
        int res = 0;
        for (char c : s.toCharArray()) {
            if (c == 'b') {
                ++bCount;
            } else if (bCount != 0) {
                ++res;
                --bCount;
            }
        }
        return res;

    }

    // 451. 根据字符出现频率排序
    public String frequencySort(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        List<Map.Entry<Character, Integer>> list = new ArrayList<>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<Character, Integer>>() {

            @Override
            public int compare(Map.Entry<Character, Integer> o1, Map.Entry<Character, Integer> o2) {
                return o2.getValue() - o1.getValue();
            }
        });
        StringBuilder res = new StringBuilder();
        for (Map.Entry<Character, Integer> item : list) {
            char key = item.getKey();
            int val = item.getValue();
            for (int i = 0; i < val; ++i) {
                res.append(key);
            }
        }
        return res.toString();

    }

    // 451. 根据字符出现频率排序
    public String frequencySort2(String s) {
        int[] counts = new int[128];
        for (char c : s.toCharArray()) {
            ++counts[c];
        }
        List<Character> list = new ArrayList<>();
        for (int i = 0; i < counts.length; ++i) {
            if (counts[i] != 0) {
                list.add((char) i);
            }
        }
        Collections.sort(list, new Comparator<Character>() {

            @Override
            public int compare(Character o1, Character o2) {
                return counts[o2] - counts[o1];
            }
        });
        StringBuilder res = new StringBuilder();
        for (char c : list) {
            for (int i = 0; i < counts[c]; ++i) {
                res.append(c);
            }
        }
        return res.toString();

    }

    // 347. 前 K 个高频元素
    public int[] topKFrequent(int[] nums, int k) {
        // key:元素 value:频率
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        List<Map.Entry<Integer, Integer>> list = new ArrayList<>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {

            @Override
            public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
                return o2.getValue() - o1.getValue();
            }

        });
        int[] res = new int[k];
        for (int i = 0; i < k; ++i) {
            res[i] = list.get(i).getKey();
        }
        return res;
    }

    // 692. 前K个高频单词
    public List<String> topKFrequent(String[] words, int k) {
        Map<String, Integer> map = new HashMap<>();
        for (String word : words) {
            map.put(word, map.getOrDefault(word, 0) + 1);
        }
        List<Map.Entry<String, Integer>> list = new ArrayList<>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<String, Integer>>() {

            @Override
            public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) {
                if (o2.getValue() == o1.getValue()) {
                    return o1.getKey().compareTo(o2.getKey());
                }
                return o2.getValue() - o1.getValue();
            }

        });
        List<String> res = new ArrayList<>();
        for (int i = 0; i < k; ++i) {
            res.add(list.get(i).getKey());
        }
        return res;

    }

    // 面试题 17.07. 婴儿名字
    public String[] trulyMostPopular(String[] names, String[] synonyms) {
        Map<String, Integer> nameToIndex = new HashMap<>();
        Map<String, Integer> nameToFreq = new HashMap<>();
        int i = 0;
        while (i < names.length) {
            int split = names[i].indexOf("(");
            String name = names[i].substring(0, split);
            int count = Integer.parseInt(names[i].substring(split + 1, names[i].length() - 1));
            nameToIndex.put(name, i);
            nameToFreq.put(name, count);
            ++i;
        }
        for (String synonym : synonyms) {
            int split = synonym.indexOf(",");
            String s1 = synonym.substring(1, split);
            String s2 = synonym.substring(split + 1, synonym.length() - 1);
            if (!nameToIndex.containsKey(s1)) {
                nameToIndex.put(s1, i++);
                nameToFreq.put(s1, 0);
            }
            if (!nameToIndex.containsKey(s2)) {
                nameToIndex.put(s2, i++);
                nameToFreq.put(s2, 0);
            }
        }
        Union17_07 union = new Union17_07(i);

        for (String synonym : synonyms) {
            int split = synonym.indexOf(",");
            String s1 = synonym.substring(1, split);
            String s2 = synonym.substring(split + 1, synonym.length() - 1);
            union.union(nameToIndex.get(s1), nameToIndex.get(s2));

        }
        // key:rootIndex val:name列表
        Map<Integer, List<String>> rootToNames = new HashMap<>();
        for (Map.Entry<String, Integer> entry : nameToIndex.entrySet()) {
            String name = entry.getKey();
            int index = entry.getValue();
            int root = union.getRoot(index);
            rootToNames.computeIfAbsent(root, k -> new ArrayList<>()).add(name);
        }
        if (rootToNames.isEmpty()) {
            return new String[] {};
        }
        String[] res = new String[rootToNames.size()];
        int index = 0;
        for (List<String> list : rootToNames.values()) {
            int minNameIndex = 0;
            int count = 0;
            for (int j = 0; j < list.size(); ++j) {
                if (list.get(j).compareTo(list.get(minNameIndex)) < 0) {
                    minNameIndex = j;
                }
                count += nameToFreq.get(list.get(j));
            }
            StringBuilder subRes = new StringBuilder();
            subRes.append(list.get(minNameIndex)).append("(").append(count).append(")");
            res[index++] = subRes.toString();
        }
        return res;

    }

    public class Union17_07 {
        private int[] rank;
        private int[] parent;

        public Union17_07(int n) {
            rank = new int[n];
            parent = new int[n];
            Arrays.fill(rank, 1);
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
}
