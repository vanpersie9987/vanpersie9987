import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.stream.Collectors;

@SuppressWarnings("unchecked")
public class LeetCodeText {
    // private int[] nums;
    // private int target;
    // private int[] nums2;
    // private int target2;
    // private String word;
    // private char[][] board;
    // private boolean[][] marked;
    // private final int[][] directions = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 }
    // };
    // private static String[] transactions = { "alice,20,800,mtv",
    // "alice,50,100,beijing" };

    public static void main(final String[] args) {
        // int[] nums = new int[5];
        // System.out.println(majorityElement(nums));

        // int[] arr = { 3, 2, 4, 1 };
        // pancakeSort(arr);

        // simplifyPath("/a/../../b/../c//.//");
        // reverseWords("a good example ");
        // String s = "2001:0db8:85a3:0:0:8A2E:0370:7334:";
        // validIPAddress(s);
        // final List<String> list = new ArrayList<>();
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
        // boolean a = reorderedPowerOf2(10);
        // boolean i = areSentencesSimilar("My name is Haley", "My Haley");
        // int res = numSplits("aaaaa");

        // int[][] grid = {
        // { 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0 },
        // { 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1 },
        // { 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1 },
        // { 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0 },
        // { 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1 },
        // { 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0 },
        // { 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0 },
        // { 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0 },
        // { 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1 },
        // { 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0 }
        // };
        // int[][] grid2 = {
        // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        // { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
        // { 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0 },
        // { 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0 },
        // { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0 },
        // { 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0 },
        // { 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0 },
        // { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
        // { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
        // };
        // String s =
        // "RULDDLLDLRDUUUURULRURRRRLRULRLULLLRRULULDDRDLRULDRRULLUDDURDLRRUDRUDDURLLLUUDULRUDRLURRDRLLDDLLLDLRLLRUUDUURDRLDUDRUDRLUDULRLUDRLDDUULDDLDURULUDUUDDRRDUURRLRDLDLRLLDRRUUURDLULLURRRRDRRURDUURDLRRUULRURRUULULUUDURUDLRDDDDDURRRLRUDRUULUUUULDURDRULLRRRUDDDUUULUURRDRDDRLLDRLDULDLUUDRDLULLDLDDRUUUUDDRRRDLLLLURUURLRUUULRDDULUULUURDURDDDRRURLURDLLLRLULRDLDDLRDRRRRLUURRRRLDUDLLRUDLDRDLDRUULDRDULRULRRDLDLLLUDLDLULLDURUURRLLULUURLRLRDUDULLDURRUDDLDDLLUDURLLRLDLDUDLURLLDRRURRDUDLDUULDUDRRUDULLUUDURRRURLULDDLRRURULUURURRDULUULDDDUUDRLDDRLULDUDDLLLDLDURDLRLUURDDRLUDRLUDLRRLUUULLDUUDUDURRUULLDDUDLURRDDLURLDRDRUDRLDDLDULDRULUDRRDRLLUURULURRRUDRLLUURULURRLUULRDDDRDDLDRLDRLDUDRLDRLDDLDUDDURUDUDDDLRRDLUUUDUDURLRDRURUDUDDRDRRLUDURULDULDDRLDLUURUULUDRLRLRLLLLRLDRURRRUULRDURDRRDDURULLRDUDRLULRRLLLDRLRLRRDULDDUDUURLRULUUUULURULDLDRDRLDDLRLURRUULRRLDULLUULUDUDRLDUDRDLLDULURLUDDUURULDURRUURLRDRRRLDDULLLLDDRRLRRDRDLRUDUUDLRLDRDRURULDLULRRDLLURDLLDLRDRURLRUDURDRRRULURDRURLDRRRDUDUDUDURUUUUULURDUDDRRDULRDDLULRDRULDRUURRURLUDDDDLDRLDLLLLRLDRLRDRRRLLDRDRUULURLDRULLDRRDUUDLURLLDULDUUDLRRRDDUDRLDULRDLLULRRUURRRURLRRLDDUDDLULRUDULDULRDUDRLRDULRUUDDRUURUDLDRDUDDUULLUDDLLRLURURLRRULLDDDLURDRRDLLLLULLDLUDDLURLLDDRLDLLDDRDRDDUDLDURLUUUUUDLLLRLDULDDRDDDDRUDLULDRRLLLDUUUDDDRDDLLULUULRRULRUDRURDDULURDRRURUULDDDDUULLLURRRRDLDDLRLDDDRLUUDRDDRDDLUDLUUULLDLRDLURRRLRDRLURUURLULLLLRDDLLLLRUDURRLDURULURULDDRULUDRLDRLLURURRRDURURDRRUDLDDLLRRDRDDLRLRLUDUDRRUDLLDUURUURRDUDLRRLRURUDURDLRRULLDLLUDURUDDRUDULLDUDRRDDUDLLLDLRDRUURLLDLDRDDLDLLUDRDDRUUUDDULRUULRDRUDUURRRURUDLURLRDDLUULRDULRDURLLRDDDRRUDDUDUDLLDDRRUUDURDLLUURDLRULULDULRUURUDRULDRDULLULRRDDLDRDLLLDULRRDDLDRDLLRDDRLUUULUURULRULRUDULRULRUURUDUUDLDUDUUURLLURDDDUDUDLRLULDLDUDUULULLRDUDLDRUDRUULRURDDLDDRDULRLRLRRRRLRULDLLLDDRLUDLULLUUDLDRRLUDULRDRLLRRRULRLRLLUDRUUDUDDLRLDRDDDDRDLDRURULULRUURLRDLLDDRLLRUDRRDDRDUDULRUDULURRUDRDLRDUUDDLDRUDLLDDLRLULLLRUUDRRRRUULLRLLULURLDUDDURLRULULDLDRURDRLLURRDLURRURLULDLRLDUDLULLLDRDLULDLRULLLUDUDUDUDLDDDDDRDLUDUULLUDRLUURDRLULD";
        // boolean b = judgeCircle(s);

    }

    // 1. 两数之和 (Two Sum)
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            if (map.containsKey(target - nums[i])) {
                return new int[] { map.get(target - nums[i]), i };
            }
            map.put(nums[i], i);
        }
        return new int[0];

    }

    // 11. 盛最多水的容器 (Container With Most Water)
    public int maxArea(int[] height) {
        int left = 0;
        int right = height.length - 1;
        int res = 0;
        while (left < right) {
            res = Math.max(res, (right - left) * Math.min(height[left], height[right]));
            if (height[left] < height[right]) {
                ++left;
            } else {
                --right;
            }
        }
        return res;

    }

    // 15. 三数之和 (3Sum)
    // 剑指 Offer II 007. 数组中和为 0 的三个数
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i < n - 2; ++i) {
            if (nums[i] > 0) {
                break;
            }
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            if (nums[i] + nums[i + 1] + nums[i + 2] > 0) {
                break;
            }
            if (nums[i] + nums[n - 1] + nums[n - 2] < 0) {
                continue;
            }
            int j = i + 1;
            int k = n - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == 0) {
                    res.add(List.of(nums[i], nums[j], nums[k]));
                    while (j + 1 < k && nums[j + 1] == nums[j]) {
                        ++j;
                    }
                    while (k - 1 > j && nums[k] == nums[k - 1]) {
                        --k;
                    }
                    ++j;
                    --k;
                } else if (sum < 0) {
                    ++j;
                } else {
                    --k;
                }
            }
        }
        return res;

    }

    // 16. 最接近的三数之和 (3Sum Closest)
    public int threeSumClosest(int[] nums, int target) {
        Arrays.sort(nums);
        int n = nums.length;
        int min = (int) 1e9;
        for (int i = 0; i < n; ++i) {
            int j = i + 1;
            int k = n - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum == target) {
                    return sum;
                }
                if (Math.abs(sum - target) < Math.abs(min - target)) {
                    min = sum;
                }
                if (sum < target) {
                    ++j;
                } else {
                    --k;
                }
            }
        }
        return min;
    }

    // 18. 四数之和 (4Sum)
    public List<List<Integer>> fourSum(int[] nums, int target) {
        Arrays.sort(nums);
        List<List<Integer>> list = new ArrayList<>();
        for (int i = 0; i < nums.length - 3; ++i) {
            if (i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }
            if ((long) nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) {
                break;
            }
            if ((long) nums[i] + nums[nums.length - 1] + nums[nums.length - 2] + nums[nums.length - 3] < target) {
                continue;
            }
            for (int j = i + 1; j < nums.length - 2; ++j) {
                if (j > i + 1 && nums[j] == nums[j - 1]) {
                    continue;
                }
                if ((long) nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target) {
                    break;
                }
                if ((long) nums[i] + nums[j] + nums[nums.length - 1] + nums[nums.length - 2] < target) {
                    continue;
                }
                int left = j + 1;
                int right = nums.length - 1;
                while (left < right) {
                    long sum = (long) nums[i] + nums[j] + nums[left] + nums[right];
                    if (sum == target) {
                        list.add(Arrays.asList(nums[i], nums[j], nums[left], nums[right]));
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
        return list;

    }

    // 26. 删除有序数组中的重复项 (Remove Duplicates from Sorted Array)
    public int removeDuplicates(int[] nums) {
        int i = 0;
        int j = 0;
        int n = nums.length;
        while (j < n) {
            int x = nums[j];
            while (j < n && nums[j] == x) {
                ++j;
            }
            nums[i++] = x;
        }
        return i;

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

    // 27. 移除元素 (Remove Element)
    public int removeElement(int[] nums, int val) {
        int i = 0;
        int j = 0;
        int n = nums.length;
        while (j < n) {
            while (j < n && nums[j] == val) {
                ++j;
            }
            if (j < n) {
                nums[i++] = nums[j++];
            }
        }
        return i;

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
    // 剑指 Offer II 068. 查找插入位置
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

    // 42. 接雨水 (Trapping Rain Water)
    // 面试题 17.21. 直方图的水量
    public int trap(int[] height) {
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

    // 42. 接雨水 (Trapping Rain Water)
    // 面试题 17.21. 直方图的水量
    public int trap2(int[] height) {
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

    // 42. 接雨水 (Trapping Rain Water) --双指针
    // 面试题 17.21. 直方图的水量
    public int trap3(int[] height) {
        int res = 0;
        int preMax = 0;
        int sufMax = 0;
        int i = 0;
        int j = height.length - 1;
        while (i <= j) {
            preMax = Math.max(preMax, height[i]);
            sufMax = Math.max(sufMax, height[j]);
            if (preMax < sufMax) {
                res += preMax - height[i++];
            } else {
                res += sufMax - height[j--];
            }
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
    // 剑指 Offer 42. 连续子数组的最大和
    // 面试题 16.17. 连续数列
    public int maxSubArray(int[] nums) {
        int n = nums.length;
        int min = 0;
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < n; ++i) {
            if (i > 0) {
                nums[i] += nums[i - 1];
            }
            res = Math.max(res, nums[i] - min);
            min = Math.min(min, nums[i]);

        }
        return res;

    }

    // 54. 螺旋矩阵
    public List<Integer> spiralOrder(int[][] matrix) {
        int r1 = 0;
        int r2 = matrix.length - 1;
        int c1 = 0;
        int c2 = matrix[0].length - 1;
        List<Integer> res = new ArrayList<>();
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

    // 55. 跳跃游戏 (Jump Game)
    public boolean canJump(final int[] nums) {
        int lastPos = nums.length - 1;
        for (int i = nums.length - 2; i >= 0; --i) {
            if (nums[i] + i >= lastPos) {
                lastPos = i;
            }
        }
        return lastPos == 0;

    }

    // 55. 跳跃游戏 (Jump Game)
    public boolean canJump2(int[] nums) {
        int right = 0;
        int cur = 0;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            right = Math.max(right, i + nums[i]);
            if (i == cur) {
                cur = right;
            }
            if (cur >= n - 1) {
                return true;
            }
        }
        return false;

    }

    // 45. 跳跃游戏 II (Jump Game II)
    public int jump2(int[] nums) {
        int n = nums.length;
        int right = 0;
        int cur = 0;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            right = Math.max(right, i + nums[i]);
            if (cur >= n - 1) {
                return res;
            }
            if (cur == i) {
                cur = right;
                ++res;
            }
        }
        return -1;
    }

    // 1748. 唯一元素的和 (Sum of Unique Elements)
    public int sumOfUnique(int[] nums) {
        int[] counts = new int[101];
        for (int num : nums) {
            ++counts[num];
        }
        int res = 0;
        for (int i = 0; i < counts.length; ++i) {
            if (counts[i] == 1) {
                res += i;
            }
        }
        return res;

    }

    // 1748. 唯一元素的和 (Sum of Unique Elements)
    public int sumOfUnique2(int[] nums) {
        if (nums.length == 1) {
            return nums[0];
        }
        int res = 0;
        Arrays.sort(nums);
        for (int i = 0; i < nums.length; ++i) {
            if (i == 0) {
                if (i + 1 < nums.length && nums[i] != nums[i + 1]) {
                    res += nums[i];
                }
            } else if (i == nums.length - 1) {
                if (nums[i] != nums[i - 1]) {
                    res += nums[i];
                }
            } else {
                if (nums[i - 1] != nums[i] && nums[i] != nums[i + 1]) {
                    res += nums[i];
                }
            }

        }
        return res;

    }

    // 57. 插入区间 (Insert Interval)
    public int[][] insert(int[][] intervals, int[] newInterval) {
        int n = intervals.length;
        List<int[]> list = new ArrayList<>();
        int i = 0;
        boolean f = false;
        while (i < n) {
            int l = intervals[i][0];
            int r = intervals[i][1];
            int L = newInterval[0];
            int R = newInterval[1];
            // 没有交集
            if (r < L) {
                list.add(intervals[i]);
                ++i;
            } else if (R < l) {
                f = true;
                list.add(newInterval);
                while (i < n) {
                    list.add(intervals[i]);
                    ++i;
                }
            } else {
                f = true;
                int min = Math.min(l, L);
                int max = Math.max(r, R);
                ++i;
                while (i < n && intervals[i][0] <= max) {
                    max = Math.max(max, intervals[i][1]);
                    ++i;
                }
                list.add(new int[] { min, max });
                while (i < n) {
                    list.add(intervals[i]);
                    ++i;
                }
            }
        }
        if (!f) {
            list.add(newInterval);
        }
        return list.toArray(new int[0][]);

    }

    // 59. 螺旋矩阵 II (Spiral Matrix II)
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int[][] dirs = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        int r = 0;
        int c = 0;
        int d = 0;
        for (int i = 1; i <= n * n; ++i) {
            res[r][c] = i;
            if (r + dirs[d][0] < 0 || r + dirs[d][0] == n || c + dirs[d][1] < 0 || c + dirs[d][1] == n
                    || res[r + dirs[d][0]][c + dirs[d][1]] != 0) {
                d = (d + 1) % 4;
            }
            r += dirs[d][0];
            c += dirs[d][1];
        }
        return res;

    }

    // 62. 不同路径 (Unique Paths) --记忆化搜索
    // 剑指 Offer II 098. 路径的数目 m行 n列
    private int[][] memo62;
    private int m62;
    private int n62;

    public int uniquePaths(int m, int n) {
        this.memo62 = new int[m][n];
        this.m62 = m;
        this.n62 = n;
        return dfs62(0, 0);

    }

    private int dfs62(int i, int j) {
        if (i == m62 || j == n62) {
            return 0;
        }
        if (i == m62 - 1 && j == n62 - 1) {
            return 1;
        }
        if (memo62[i][j] != 0) {
            return memo62[i][j];
        }
        return memo62[i][j] = dfs62(i + 1, j) + dfs62(i, j + 1);
    }

    // 62. 不同路径 (Unique Paths)
    // 剑指 Offer II 098. 路径的数目 m行 n列
    public int uniquePaths2(final int m, final int n) {
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

    // 63. 不同路径 II (Unique Paths II) --记忆化搜索
    private int m63;
    private int n63;
    private int[][] grid63;
    private int[][] memo63;

    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        this.m63 = obstacleGrid.length;
        this.n63 = obstacleGrid[0].length;
        this.grid63 = obstacleGrid;
        this.memo63 = new int[m63][n63];
        for (int i = 0; i < m63; ++i) {
            Arrays.fill(memo63[i], -1);
        }
        return dfs63(0, 0);

    }

    private int dfs63(int i, int j) {
        if (i == m63 || j == n63 || grid63[i][j] == 1) {
            return 0;
        }
        if (i == m63 - 1 && j == n63 - 1) {
            return 1;
        }
        if (memo63[i][j] != -1) {
            return memo63[i][j];
        }
        return memo63[i][j] = dfs63(i + 1, j) + dfs63(i, j + 1);
    }

    // 63. 不同路径 II (Unique Paths II)
    public int uniquePathsWithObstacles2(final int[][] obstacleGrid) {
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

    // 64. 最小路径和 (Minimum Path Sum)
    // 剑指 Offer II 099. 最小路径之和
    private int m64;
    private int n64;
    private int[][] grid64;
    private int[][] memo64;

    public int minPathSum(int[][] grid) {
        this.m64 = grid.length;
        this.n64 = grid[0].length;
        this.grid64 = grid;
        this.memo64 = new int[m64][n64];
        for (int i = 0; i < m64; ++i) {
            Arrays.fill(memo64[i], -1);
        }
        return dfs64(0, 0);

    }

    private int dfs64(int i, int j) {
        if (i == m64 || j == n64) {
            return (int) 1e5;
        }
        if (i == m64 - 1 && j == n64 - 1) {
            return grid64[i][j];
        }
        if (memo64[i][j] != -1) {
            return memo64[i][j];
        }
        return memo64[i][j] = Math.min(dfs64(i + 1, j), dfs64(i, j + 1)) + grid64[i][j];
    }

    // 64. 最小路径和 (Minimum Path Sum)
    // 剑指 Offer II 099. 最小路径之和
    public int minPathSum2(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 1; i < n; ++i) {
            grid[0][i] += grid[0][i - 1];
        }
        for (int i = 1; i < m; ++i) {
            grid[i][0] += grid[i - 1][0];
        }
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        return grid[m - 1][n - 1];

    }

    // 73. 矩阵置零 / 面试题 01.08. 零矩阵
    public void setZeroes(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;
        boolean firstCol0 = false;
        boolean firstRow0 = false;
        for (int i = 0; i < m; ++i) {
            if (matrix[i][0] == 0) {
                firstCol0 = true;
                break;
            }
        }
        for (int j = 0; j < n; ++j) {
            if (matrix[0][j] == 0) {
                firstRow0 = true;
                break;
            }
        }

        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (matrix[i][j] == 0) {
                    matrix[0][j] = matrix[i][0] = 0;
                }
            }
        }

        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }

            }
        }

        if (firstCol0) {
            for (int i = 0; i < m; ++i) {
                matrix[i][0] = 0;
            }
        }

        if (firstRow0) {
            for (int j = 0; j < n; ++j) {
                matrix[0][j] = 0;
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

    // 79. 单词搜索 (Word Search) --回溯
    // LCR 129. 字母迷宫
    private int m79;
    private int n79;
    private int l79;
    private char[][] board79;
    private String word79;
    private boolean[][] vis79;

    public boolean exist(char[][] board, String word) {
        this.m79 = board.length;
        this.n79 = board[0].length;
        this.l79 = word.length();
        this.board79 = board;
        this.word79 = word;
        this.vis79 = new boolean[m79][n79];
        for (int i = 0; i < m79; ++i) {
            for (int j = 0; j < n79; ++j) {
                if (dfs79(i, j, 0)) {
                    return true;
                }
            }
        }
        return false;

    }

    private boolean dfs79(int i, int j, int k) {
        if (k == l79) {
            return true;
        }
        if (!(i >= 0 && i < m79 && j >= 0 && j < n79)) {
            return false;
        }
        if (vis79[i][j]) {
            return false;
        }
        if (board79[i][j] != word79.charAt(k)) {
            return false;
        }
        vis79[i][j] = true;
        int[][] dirs = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 } };
        for (int[] d : dirs) {
            if (dfs79(i + d[0], j + d[1], k + 1)) {
                return true;
            }
        }
        vis79[i][j] = false;
        return false;
    }

    // 1295. 统计位数为偶数的数字 (Find Numbers with Even Number of Digits)
    public int findNumbers(int[] nums) {
        int count = 0;
        for (int num : nums) {
            if (((int) (Math.log10(num) + 1)) % 2 == 0) {
                ++count;
            }
        }
        return count;

    }

    // 88. 合并两个有序数组 (Merge Sorted Array)
    // 面试题 10.01. 合并排序的数组
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int p = m + n - 1;
        while (i >= 0 && j >= 0) {
            if (nums1[i] > nums2[j]) {
                nums1[p--] = nums1[i--];
            } else {
                nums1[p--] = nums2[j--];
            }
        }
        while (j >= 0) {
            nums1[p--] = nums2[j--];
        }
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

    // 120. 三角形最小路径和 (Triangle)
    private int n120;
    private List<List<Integer>> triangle120;
    private int[][] memo120;

    public int minimumTotal(List<List<Integer>> triangle) {
        this.n120 = triangle.size();
        this.triangle120 = triangle;
        this.memo120 = new int[n120][n120];
        for (int i = 0; i < n120; ++i) {
            Arrays.fill(memo120[i], (int) 1e7);
        }
        return dfs120(0, 0);

    }

    private int dfs120(int i, int j) {
        if (i == n120) {
            return 0;
        }
        if (memo120[i][j] != (int) 1e7) {
            return memo120[i][j];
        }
        return memo120[i][j] = Math.min(dfs120(i + 1, j), dfs120(i + 1, j + 1)) + triangle120.get(i).get(j);
    }

    // 120. 三角形最小路径和 (Triangle)
    // 剑指 Offer II 100. 三角形中最小路径之和 --动态规划
    public int minimumTotal2(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[] dp = new int[n + 1];
        for (int i = n - 1; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                dp[j] = Math.min(dp[j], dp[j + 1]) + triangle.get(i).get(j);
            }
        }
        return dp[0];

    }

    // 121. 买卖股票的最佳时机 (Best Time to Buy and Sell Stock)
    // 剑指 Offer 63. 股票的最大利润
    public int maxProfit121(int[] prices) {
        int res = 0;
        int min = Integer.MAX_VALUE;
        for (int price : prices) {
            if (price < min) {
                min = price;
            } else if (price - min > res) {
                res = price - min;
            }
        }
        return res;
    }

    // 121. 买卖股票的最佳时机 (Best Time to Buy and Sell Stock)
    // 剑指 Offer 63. 股票的最大利润
    private int n121;
    private int[] prices121;
    private int[][] memo121;

    public int maxProfit121_2(int[] prices) {
        this.n121 = prices.length;
        this.prices121 = prices;
        this.memo121 = new int[n121][2];
        for (int i = 0; i < n121; ++i) {
            Arrays.fill(memo121[i], -1);
        }
        return dfs121(0, 0);

    }

    private int dfs121(int i, int state) {
        if (i == n121) {
            return 0;
        }
        if (memo121[i][state] != -1) {
            return memo121[i][state];
        }
        int max = 0;
        // 可买入
        if (state == 0) {
            // 买
            max = Math.max(max, -prices121[i] + dfs121(i + 1, 1));
            // 不买
            max = Math.max(max, dfs121(i + 1, state));
        }
        // 可卖出
        else {
            // 卖
            max = Math.max(max, prices121[i]);
            // 不卖
            max = Math.max(max, dfs121(i + 1, state));
        }
        return memo121[i][state] = max;
    }

    // 122. 买卖股票的最佳时机 II (Best Time to Buy and Sell Stock II)
    public int maxProfit122(int[] prices) {
        int max = 0;
        for (int i = 1; i < prices.length; ++i) {
            if (prices[i] > prices[i - 1]) {
                max += prices[i] - prices[i - 1];
            }
        }
        return max;

    }

    // 122. 买卖股票的最佳时机 II (Best Time to Buy and Sell Stock II)
    private int[][] memo122;
    private int[] prices122;
    private int n122;

    public int maxProfit122_2(int[] prices) {
        this.n122 = prices.length;
        this.prices122 = prices;
        this.memo122 = new int[n122][2];
        for (int i = 0; i < n122; ++i) {
            Arrays.fill(memo122[i], Integer.MIN_VALUE);
        }
        return dfs122(0, 0);

    }

    private int dfs122(int i, int state) {
        if (i == n122) {
            return 0;
        }
        if (memo122[i][state] != Integer.MIN_VALUE) {
            return memo122[i][state];
        }
        int max = Integer.MIN_VALUE;
        // 可买入
        if (state == 0) {
            // 买
            max = Math.max(max, -prices122[i] + dfs122(i + 1, state ^ 1));
            // 不买
            max = Math.max(max, dfs122(i + 1, state));
        }
        // 可卖出
        else {
            // 卖
            max = Math.max(max, prices122[i] + dfs122(i + 1, state ^ 1));
            // 不卖
            max = Math.max(max, dfs122(i + 1, state));
        }
        return memo122[i][state] = max;
    }

    // 123. 买卖股票的最佳时机 III (Best Time to Buy and Sell Stock III)
    public int maxProfit123(int[] prices) {
        int n = prices.length;
        int buy1 = prices[0], sell1 = 0;
        int buy2 = prices[0], sell2 = 0;
        for (int i = 1; i < n; ++i) {
            buy1 = Math.min(buy1, prices[i]);
            sell1 = Math.max(sell1, prices[i] - buy1);
            buy2 = Math.min(buy2, prices[i] - sell1);
            sell2 = Math.max(sell2, prices[i] - buy2);
        }
        return sell2;

    }

    // 123. 买卖股票的最佳时机 III (Best Time to Buy and Sell Stock III)
    private int[][][] memo123;
    private int n123;
    private int[] prices123;

    public int maxProfit123_2(int[] prices) {
        this.n123 = prices.length;
        this.prices123 = prices;
        this.memo123 = new int[n123][2][2];
        for (int i = 0; i < n123; ++i) {
            for (int j = 0; j < 2; ++j) {
                Arrays.fill(memo123[i][j], -1);
            }
        }
        return dfs123(0, 0, 0);

    }

    private int dfs123(int i, int count, int status) {
        if (i == n123 || count == 2) {
            return 0;
        }
        if (memo123[i][count][status] != -1) {
            return memo123[i][count][status];
        }
        int max = 0;
        // 可买入
        if (status == 0) {
            // 买
            max = Math.max(max, -prices123[i] + dfs123(i + 1, count, status ^ 1));
            // 不买
            max = Math.max(max, dfs123(i + 1, count, status));
        }
        // 可卖出
        else {
            // 卖
            max = Math.max(max, prices123[i] + dfs123(i + 1, count + 1, status ^ 1));
            // 不卖
            max = Math.max(max, dfs123(i + 1, count, status));
        }
        return memo123[i][count][status] = max;
    }

    // 714. 买卖股票的最佳时机含手续费 (Best Time to Buy and Sell Stock with Transaction Fee)
    public int maxProfit714(int[] prices, int fee) {
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

    // 714. 买卖股票的最佳时机含手续费 (Best Time to Buy and Sell Stock with Transaction Fee)
    private int[][] memo714;
    private int[] prices714;
    private int fee714;
    private int n714;

    public int maxProfit714_2(int[] prices, int fee) {
        this.n714 = prices.length;
        this.fee714 = fee;
        this.prices714 = prices;
        this.memo714 = new int[n714][2];
        for (int i = 0; i < n714; ++i) {
            Arrays.fill(memo714[i], -1);
        }
        return dfs714(0, 0);

    }

    private int dfs714(int i, int state) {
        if (i == n714) {
            return 0;
        }
        if (memo714[i][state] != -1) {
            return memo714[i][state];
        }
        int max = 0;
        // 可买入
        if (state == 0) {
            // 买
            max = Math.max(max, -prices714[i] - fee714 + dfs714(i + 1, state ^ 1));
            // 不买
            max = Math.max(max, dfs714(i + 1, state));
        }
        // 可卖出
        else {
            // 卖
            max = Math.max(max, prices714[i] + dfs714(i + 1, state ^ 1));
            // 不卖
            max = Math.max(max, dfs714(i + 1, state));
        }
        return memo714[i][state] = max;
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

    // 128. 最长连续序列 // 剑指 Offer II 119. 最长连续序列
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

    // 152. 乘积最大子数组 (Maximum Product Subarray)
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

    // 152. 乘积最大子数组 (Maximum Product Subarray)
    private int[] nums152;
    private int n152;
    private int[][] memo152;

    public int maxProduct2(int[] nums) {
        this.nums152 = nums;
        this.n152 = nums.length;
        this.memo152 = new int[n152][2];
        for (int i = 0; i < n152; ++i) {
            Arrays.fill(memo152[i], (int) 1e8);
        }
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < n152; ++i) {
            res = Math.max(res, dfs152(i, 0));
        }
        return res;
    }

    // 以 i 结尾、最大 (j == 0) 子数组的乘积
    // 以 i 结尾、最小 (j == 1) 子数组的乘积
    private int dfs152(int i, int j) {
        if (i < 0) {
            return 1;
        }
        if (memo152[i][j] != (int) 1e8) {
            return memo152[i][j];
        }
        int res = nums152[i];
        if (j == 0) {
            if (nums152[i] < 0) {
                res = Math.max(res, dfs152(i - 1, j ^ 1) * nums152[i]);
            } else {
                res = Math.max(res, dfs152(i - 1, j) * nums152[i]);
            }
        } else {
            if (nums152[i] < 0) {
                res = Math.min(res, dfs152(i - 1, j ^ 1) * nums152[i]);
            } else {
                res = Math.min(res, dfs152(i - 1, j) * nums152[i]);
            }
        }
        return memo152[i][j] = res;
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

    // 154. 寻找旋转排序数组中的最小值 II (Find Minimum in Rotated Sorted Array II) --需要掌握二分查找
    // 剑指 Offer 11. 旋转数组的最小数字
    public int findMin154(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] < nums[right]) {
                right = mid;
            } else if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else {
                --right;
            }
        }
        return nums[left];
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

    // 剑指 Offer 57. 和为s的两个数字 // 剑指 Offer II 006. 排序数组中两个数字之和
    public int[] twoSumOffer_57(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int sum = nums[left] + nums[right];
            if (sum == target) {
                return new int[] { nums[left], nums[right] };
            }
            if (sum < target) {
                ++left;
            } else {
                --right;
            }
        }
        return null;

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

    // 228. 汇总区间 (Summary Ranges)
    public List<String> summaryRanges(int[] nums) {
        int n = nums.length;
        int i = 0;
        List<String> res = new ArrayList<>();
        while (i < n) {
            StringBuilder cur = new StringBuilder();
            cur.append(nums[i]);
            int j = i + 1;
            while (j < n && nums[j] - nums[j - 1] == 1) {
                ++j;
            }
            if (j - i != 1) {
                cur.append("->").append(nums[j - 1]);
            }
            res.add(cur.toString());
            i = j;
        }
        return res;

    }

    // 229. 求众数 II
    public List<Integer> majorityElement2(final int[] nums) {
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
    public List<Integer> majorityElement3(int[] nums) {
        int majorityA = nums[0];
        int countA = 0;
        int majorityB = nums[0];
        int countB = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (majorityA == nums[i]) {
                ++countA;
            } else if (majorityB == nums[i]) {
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
        for (int num : nums) {
            if (num == majorityA) {
                ++countA;
            } else if (num == majorityB) {
                ++countB;
            }
        }
        List<Integer> res = new ArrayList<>();
        if (countA * 3 > nums.length) {
            res.add(majorityA);
        }
        if (countB * 3 > nums.length) {
            res.add(majorityB);
        }
        return res;

    }

    // 268. 丢失的数字
    public int missingNumber(final int[] nums) {
        int missing = nums.length;
        for (int i = 0; i < nums.length; ++i) {
            missing ^= i * nums[i];
        }
        return missing;

    }

    // 268. 丢失的数字
    public int missingNumber2(final int[] nums) {
        final int n = nums.length;
        final int sumExpect = (0 + n) * (n + 1) / 2;
        int sumReal = Arrays.stream(nums).sum();
        return sumExpect - sumReal;

    }

    // 283. 移动零
    public void moveZeroes(final int[] nums) {
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
    public int findDuplicate(final int[] nums) {
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

    class ListNode {
        public int val;
        public ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }

        ListNode(int x, ListNode next) {
            this.val = x;
            this.next = next;
        }
    }

    // 289. 生命游戏
    public void gameOfLife(int[][] board) {
        // 1 --> 1 1
        // 0 --> 0 0
        // 1 --> 0 3
        // 0 --> 1 2

        int m = board.length;
        int n = board[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                getTransfer289(board, i, j);
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 0 || board[i][j] == 3) {
                    board[i][j] = 0;
                } else {
                    board[i][j] = 1;
                }
            }
        }

    }

    private void getTransfer289(int[][] board, int i, int j) {
        int count = 0;
        int top = Math.max(i - 1, 0);
        int bottom = Math.min(i + 1, board.length - 1);
        int left = Math.max(j - 1, 0);
        int right = Math.min(j + 1, board[0].length - 1);
        for (int x = top; x <= bottom; ++x) {
            for (int y = left; y <= right; ++y) {
                if (board[x][y] == 1 || board[x][y] == 3) {
                    ++count;
                }
            }
        }
        if (board[i][j] == 1) {
            if (count == 3 || count == 4) {
                board[i][j] = 1;
            } else {
                board[i][j] = 3;
            }
        } else {
            if (count == 3) {
                board[i][j] = 2;
            } else {
                board[i][j] = 0;
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

    // 495. 提莫攻击
    public int findPoisonedDuration(int[] timeSeries, int duration) {
        int res = 0;
        for (int i = 1; i < timeSeries.length; ++i) {
            res += Math.min(duration, timeSeries[i] - timeSeries[i - 1]);
        }
        return res + duration;

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

    // 565. 数组嵌套 (Array Nesting)
    public int arrayNesting(int[] nums) {
        int n = nums.length;
        int res = 0;
        boolean[] visited = new boolean[n];
        for (int i = 0; i < n; ++i) {
            int count = 0;
            int index = i;
            while (!visited[index]) {
                ++count;
                visited[index] = true;
                index = nums[index];
            }
            res = Math.max(res, count);
        }
        return res;
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
    public int[][] matrixReshape(int[][] mat, int r, int c) {
        if (mat == null || mat.length == 0 || mat[0].length == 0) {
            return null;
        }
        if (mat.length * mat[0].length != r * c) {
            return mat;
        }
        int index = 0;
        int[][] res = new int[r][c];
        for (int i = 0; i < mat.length; ++i) {
            for (int j = 0; j < mat[0].length; ++j) {
                res[index / c][index % c] = mat[i][j];
                ++index;
            }
        }
        return res;

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

    // 1684. 统计一致字符串的数目 (Count the Number of Consistent Strings)
    public int countConsistentStrings(String allowed, String[] words) {
        int mask = 0;
        for (char c : allowed.toCharArray()) {
            mask |= 1 << (c - 'a');
        }
        int res = 0;
        search: for (String word : words) {
            for (char c : word.toCharArray()) {
                int cur = 1 << (c - 'a');
                if ((cur & mask) == 0) {
                    continue search;
                }
            }
            ++res;
        }
        return res;

    }

    // 605. 种花问题 (Can Place Flowers)
    public boolean canPlaceFlowers(int[] flowerbed, int n) {
        List<Integer> list = Arrays.stream(flowerbed).boxed().collect(Collectors.toList());
        list.add(0, 0);
        list.add(0);
        for (int i = 1; i < list.size() - 1; ++i) {
            if (list.get(i - 1) == 0 && list.get(i) == 0 && list.get(i + 1) == 0) {
                --n;
                list.set(i, 1);
            }
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

    // 621. 任务调度器 (Task Scheduler)
    public int leastInterval(char[] tasks, int n) {
        int[] counts = new int[26];
        for (char task : tasks) {
            ++counts[task - 'A'];
        }
        Arrays.sort(counts);
        int max = counts[25] - 1;
        int slots = max * n;
        for (int i = counts.length - 2; i >= 0; --i) {
            slots -= Math.min(max, counts[i]);
        }
        return slots > 0 ? tasks.length + slots : tasks.length;
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

    // 665. 非递减数列 (Non-decreasing Array)
    public boolean checkPossibility(int[] nums) {
        int n = nums.length;
        int count = 0;
        for (int i = 0; i < n - 1; ++i) {
            int x = nums[i];
            int y = nums[i + 1];
            if (x > y) {
                if (++count > 1) {
                    return false;
                }
                if (i > 0 && y < nums[i - 1]) {
                    nums[i + 1] = x;
                }
            }
        }
        return true;

    }

    // n = 8
    // k == 1 // 1 2 3 4 5 6 7 8
    // k == 2 // 1 8 7 6 5 4 3 2
    // k == 3 // 1 8 2 3 4 5 6 7

    // 667. 优美的排列 II (Beautiful Arrangement II)
    public int[] constructArray(int n, int k) {
        int[] res = new int[n];
        for (int i = 1; i <= n; ++i) {
            res[i - 1] = i;
        }
        for (int i = 1; i < k; ++i) {
            reverse667(res, i);

        }
        return res;

    }

    private void reverse667(int[] res, int i) {
        int left = i;
        int right = res.length - 1;
        while (left < right) {
            int temp = res[left];
            res[left] = res[right];
            res[right] = temp;
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

    // 695. 岛屿的最大面积 (Max Area of Island) --dfs
    // 剑指 Offer II 105. 岛屿的最大面积
    private int[][] grid695;
    private int res695;
    private int m695;
    private int n695;

    public int maxAreaOfIsland(final int[][] grid) {
        this.grid695 = grid;
        this.m695 = grid.length;
        this.n695 = grid[0].length;
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                if (grid[i][j] == 1) {
                    res695 = Math.max(res695, dfs695(i, j));
                }
            }
        }
        return res695;

    }

    private int dfs695(int i, int j) {
        if (!(i >= 0 && i < m695 && j >= 0 && j < n695)) {
            return 0;
        }
        if (grid695[i][j] == 0) {
            return 0;
        }
        grid695[i][j] = 0;
        int res = 1;
        res += dfs695(i + 1, j);
        res += dfs695(i - 1, j);
        res += dfs695(i, j - 1);
        res += dfs695(i, j + 1);
        return res;
    }

    // 695. 岛屿的最大面积 (Max Area of Island) --并查集
    // 剑指 Offer II 105. 岛屿的最大面积
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

    // 695. 岛屿的最大面积 (Max Area of Island) --bfs
    // 剑指 Offer II 105. 岛屿的最大面积
    public int maxAreaOfIsland3(int[][] grid) {
        int res = 0;
        int m = grid.length;
        int n = grid[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    res = Math.max(res, maxAreaWithBFS695(grid, i, j));
                }
            }
        }
        return res;

    }

    private int maxAreaWithBFS695(int[][] grid, int i, int j) {
        int res = 0;
        int m = grid.length;
        int n = grid[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { i, j });
        grid[i][j] = 0;
        while (!queue.isEmpty()) {
            ++res;
            int[] cur = queue.poll();
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                    grid[nx][ny] = 0;
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        return res;

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

    // 717. 1比特与2比特字符 (1-bit and 2-bit Characters)
    public boolean isOneBitCharacter(int[] bits) {
        int i = 0;
        int n = bits.length;
        while (i < n - 1) {
            i += bits[i] + 1;
        }
        return i == n - 1;

    }

    // 717. 1比特与2比特字符 (1-bit and 2-bit Characters)
    public boolean isOneBitCharacter2(final int[] bits) {
        int cnt1 = 0;
        for (int i = bits.length - 2; i >= 0 && bits[i] == 1; --i) {
            ++cnt1;
        }
        return cnt1 % 2 == 0;

    }

    // 729. 我的日程安排表 I (My Calendar I) --还需掌握线段树
    // 剑指 Offer II 058. 日程表
    class MyCalendar {
        List<int[]> list;

        public MyCalendar() {
            list = new ArrayList<>();
        }

        public boolean book(int start, int end) {
            for (int[] cur : list) {
                if (!(end <= cur[0] || start >= cur[1])) {
                    return false;
                }
            }
            list.add(new int[] { start, end });
            return true;
        }
    }

    // 746. 使用最小花费爬楼梯 (Min Cost Climbing Stairs)
    // 剑指 Offer II 088. 爬楼梯的最少成本
    private int[] cost746;
    private int[] memo746;

    public int minCostClimbingStairs(int[] cost) {
        int n = cost.length;
        this.cost746 = cost;
        this.memo746 = new int[n];
        return Math.min(dfs746(n - 1), dfs746(n - 2));

    }

    private int dfs746(int i) {
        if (i < 0) {
            return 0;
        }
        if (i <= 1) {
            return cost746[i];
        }
        if (memo746[i] > 0) {
            return memo746[i];
        }
        return memo746[i] = Math.min(dfs746(i - 2), dfs746(i - 1)) + cost746[i];
    }

    // 746. 使用最小花费爬楼梯 (Min Cost Climbing Stairs)
    // 剑指 Offer II 088. 爬楼梯的最少成本
    public int minCostClimbingStairs2(int[] cost) {
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

    // 769. 最多能完成排序的块 (Max Chunks To Make Sorted) --贪心
    public int maxChunksToSorted(int[] arr) {
        int n = arr.length;
        int max = 0;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            max = Math.max(max, arr[i]);
            if (i == max) {
                ++res;
            }
        }
        return res;

    }

    // 768. 最多能完成排序的块 II (Max Chunks To Make Sorted II) --哈希表
    public int maxChunksToSorted768(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        int n = arr.length;
        int[] sortedArr = Arrays.copyOf(arr, n);
        Arrays.sort(sortedArr);
        for (int i = 0; i < n; ++i) {
            map.put(arr[i], map.getOrDefault(arr[i], 0) + 1);
            if (map.get(arr[i]) == 0) {
                map.remove(arr[i]);
            }
            map.put(sortedArr[i], map.getOrDefault(sortedArr[i], 0) - 1);
            if (map.get(sortedArr[i]) == 0) {
                map.remove(sortedArr[i]);
            }
            if (map.isEmpty()) {
                ++res;
            }
        }
        return res;

    }

    // 768. 最多能完成排序的块 II (Max Chunks To Make Sorted II)
    public int maxChunksToSorted768_2(int[] arr) {
        Stack<Integer> stack = new Stack<>();
        for (int num : arr) {
            if (!stack.isEmpty() && num < stack.peek()) {
                int top = stack.pop();
                while (!stack.isEmpty() && num < stack.peek()) {
                    stack.pop();
                }
                stack.push(top);
            } else {
                stack.push(num);
            }
        }
        return stack.size();

    }

    // 775. 全局倒置与局部倒置 (Global and Local Inversions)
    public boolean isIdealPermutation(int[] nums) {
        int n = nums.length;
        int min = nums[n - 1];
        for (int i = n - 3; i >= 0; --i) {
            if (nums[i] > min) {
                return false;
            }
            min = Math.min(min, nums[i + 1]);
        }
        return true;

    }

    // 775. 全局倒置与局部倒置 (Global and Local Inversions)
    public boolean isIdealPermutation2(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (Math.abs(nums[i] - i) > 1) {
                return false;
            }
        }
        return true;

    }

    // 1779. 找到最近的有相同 X 或 Y 坐标的点 (Find Nearest Point That Has the Same X or Y
    // Coordinate)
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

    // 792. 匹配子序列的单词数 (Number of Matching Subsequences) --二分查找
    public int numMatchingSubseq(String s, String[] words) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int n = s.length();
        for (int i = 0; i < n; ++i) {
            int index = s.charAt(i) - 'a';
            map.computeIfAbsent(index, k -> new ArrayList<>()).add(i);
        }
        int res = words.length;
        for (String word : words) {
            if (word.length() > n) {
                --res;
                continue;
            }
            int p = -1;
            for (char c : word.toCharArray()) {
                int index = c - 'a';
                p = binarySearch792(map.getOrDefault(index, new ArrayList<>()), p);
                if (p == -1) {
                    --res;
                    break;
                }
            }
        }
        return res;

    }

    // 返回排序数组list中，第一个比target值大的的元素值，若不存在，返回-1
    private int binarySearch792(List<Integer> list, int target) {
        if (list.isEmpty()) {
            return -1;
        }
        int n = list.size();
        if (target >= list.get(n - 1)) {
            return -1;
        }
        if (list.get(0) > target) {
            return list.get(0);
        }
        int left = 0;
        int right = list.size() - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (list.get(mid) > target) {
                res = list.get(mid);
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    // 795. 区间子数组个数 (Number of Subarrays with Bounded Maximum)
    public int numSubarrayBoundedMax(int[] nums, int left, int right) {
        return (int) (counts795(nums, right) - counts795(nums, left - 1));
    }

    // 统计nums中，子数组的最大值 <= bound的个数
    private long counts795(int[] nums, int bound) {
        int count = 0;
        long res = 0l;
        for (int num : nums) {
            if (num > bound) {
                count = 0;
            } else {
                ++count;
            }
            res += count;
        }
        return res;
    }

    // 795. 区间子数组个数 (Number of Subarrays with Bounded Maximum)
    public int numSubarrayBoundedMax2(int[] nums, int left, int right) {
        int index1 = -1;
        int index2 = -1;
        int res = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] > right) {
                index1 = i;
            }
            if (nums[i] >= left) {
                index2 = i;
            }
            res += index2 - index1;
        }
        return res;

    }

    // 825. 适龄的朋友 (Friends Of Appropriate Ages)
    public int numFriendRequests(final int[] ages) {
        int res = 0;
        int[] count = new int[121];
        for (int age : ages) {
            ++count[age];
        }
        for (int ageA = 1; ageA < 121; ++ageA) {
            if (count[ageA] == 0) {
                continue;
            }
            for (int ageB = 1; ageB < 121; ++ageB) {
                if ((count[ageB] != 0) && (!(ageB <= 0.5 * ageA + 7 || ageB > ageA || (ageB > 100) && (ageA < 100)))) {
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

    // 840. 矩阵中的幻方 (Magic Squares In Grid)
    public int numMagicSquaresInside(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = 0;
        for (int i = 0; i < m - 2; ++i) {
            for (int j = 0; j < n - 2; ++j) {
                if (grid[i + 1][j + 1] == 5 && check840(grid, i, j)) {
                    ++res;
                }
            }
        }
        return res;

    }

    private boolean check840(int[][] grid, int x0, int y0) {
        int mask = 0;
        for (int i = x0; i < x0 + 3; ++i) {
            for (int j = y0; j < y0 + 3; ++j) {
                int v = grid[i][j];
                if (v < 1 || v > 9 || ((mask >> v) & 1) != 0) {
                    return false;
                }
                mask |= 1 << v;
            }
        }
        for (int i = x0; i < x0 + 3; ++i) {
            int rowSum = 0;
            for (int j = y0; j < y0 + 3; ++j) {
                rowSum += grid[i][j];
            }
            if (rowSum != 15) {
                return false;
            }
        }
        for (int j = y0; j < y0 + 3; ++j) {
            int colSum = 0;
            for (int i = x0; i < x0 + 3; ++i) {
                colSum += grid[i][j];
            }
            if (colSum != 15) {
                return false;
            }
        }
        return true;
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

    // 面试题 01.09. 字符串轮转
    public boolean isFlipedString(String s1, String s2) {
        if (s1.equals(s2)) {
            return true;
        }
        if (s1.length() != s2.length()) {
            return false;
        }
        char[] chars1 = s1.toCharArray();
        char[] chars2 = s2.toCharArray();
        reverse2(chars1, 0, chars1.length - 1);
        for (int i = 0; i < chars1.length; ++i) {
            char[] chars1Clone = chars1.clone();
            reverse2(chars1Clone, 0, i);
            reverse2(chars1Clone, i + 1, chars1Clone.length - 1);
            if (Arrays.equals(chars1Clone, chars2)) {
                return true;
            }
        }
        return false;
    }

    private void reverse2(char[] chars1, int left, int right) {
        char temp;
        while (left < right) {
            temp = chars1[left];
            chars1[left] = chars1[right];
            chars1[right] = temp;
            ++left;
            --right;
        }
    }

    // 面试题 01.09. 字符串轮转
    public boolean isFlipedString2(String s1, String s2) {
        String s = s1 + s1;
        return s.contains(s2) && (s.length() == s2.length() * 2);

    }

    // 849. 到最近的人的最大距离 (Maximize Distance to Closest Person)
    public int maxDistToClosest(int[] seats) {
        int last = -1;
        int n = seats.length;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (seats[i] == 1) {
                if (last == -1) {
                    res = Math.max(res, i);
                } else {
                    res = Math.max(res, (i - last) / 2);
                }
                last = i;
            }
        }
        res = Math.max(res, n - 1 - last);
        return res;

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

    // 870. 优势洗牌 (Advantage Shuffle) --二分查找
    public int[] advantageCount(int[] nums1, int[] nums2) {
        int n = nums1.length;
        Arrays.sort(nums1);
        List<Integer> list = Arrays.stream(nums1).boxed().collect(Collectors.toList());
        int[] res = new int[n];
        int i = 0;
        while (i < n) {
            int index = binarySearch870(list, nums2[i]);
            res[i] = list.get(index);
            list.remove(index);
            ++i;
        }
        return res;

    }

    private int binarySearch870(List<Integer> list, int target) {
        int n = list.size();
        if (target >= list.get(n - 1)) {
            return 0;
        }
        if (target < list.get(0)) {
            return 0;
        }
        int res = 0;
        int left = 0;
        int right = n - 1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (list.get(mid) > target) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    // 870. 优势洗牌 (Advantage Shuffle) --双指针
    public int[] advantageCount2(int[] nums1, int[] nums2) {
        int n = nums1.length;
        Arrays.sort(nums1);
        Integer[] ids2 = new Integer[n];
        for (int i = 0; i < n; ++i) {
            ids2[i] = i;
        }

        Arrays.sort(ids2, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return nums2[o1] - nums2[o2];

            }

        });

        int[] res = new int[n];
        int left = 0;
        int right = n - 1;
        for (int x : nums1) {
            res[x > nums2[ids2[left]] ? ids2[left++] : ids2[right--]] = x;
        }
        return res;

    }

    // 873. 最长的斐波那契子序列的长度 (Length of Longest Fibonacci Subsequence)
    // 剑指 Offer II 093. 最长斐波那契数列
    public int lenLongestFibSubseq(int[] arr) {
        Set<Integer> set = new HashSet<>();
        for (int num : arr) {
            set.add(num);
        }
        int res = 0;
        for (int i = 0; i < arr.length; ++i) {
            for (int j = i + 1; j < arr.length; ++j) {
                int cur = 0;
                int f1 = arr[i];
                int f2 = arr[j];
                while (set.contains(f1 + f2)) {
                    int temp = f1 + f2;
                    f1 = f2;
                    f2 = temp;
                    ++cur;
                }
                res = Math.max(res, cur == 0 ? 0 : cur + 2);
            }
        }
        return res;

    }

    // 873. 最长的斐波那契子序列的长度 (Length of Longest Fibonacci Subsequence)
    // 剑指 Offer II 093. 最长斐波那契数列
    private Map<Integer, Integer> map873;
    private int[][] memo873;
    private int[] arr873;

    public int lenLongestFibSubseq2(int[] arr) {
        int n = arr.length;
        int res = 0;
        this.arr873 = arr;
        this.memo873 = new int[n][n];
        this.map873 = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            map873.put(arr[i], i);
            Arrays.fill(memo873[i], -1);
        }
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                res = Math.max(res, dfs873(i, j) + 2);
            }
        }
        return res >= 3 ? res : 0;

    }

    private int dfs873(int i, int j) {
        if (memo873[i][j] != -1) {
            return memo873[i][j];
        }
        if (!map873.containsKey(arr873[i] + arr873[j])) {
            return memo873[i][j] = 0;
        }
        return dfs873(j, map873.get(arr873[i] + arr873[j])) + 1;
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

    // 905. 按奇偶排序数组 (Sort Array By Parity)
    public int[] sortArrayByParity(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            while (left < right && nums[left] % 2 == 0) {
                ++left;
            }
            while (left < right && nums[right] % 2 == 1) {
                --right;
            }
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            ++left;
            --right;
        }
        return nums;

    }

    // 922. 按奇偶排序数组 II (Sort Array By Parity II)
    public int[] sortArrayByParityII(int[] nums) {
        int eId = 0;
        int oId = 1;
        int n = nums.length;
        int[] res = new int[n];
        for (int x : nums) {
            if ((x & 1) == 0) {
                res[eId] = x;
                eId += 2;
            } else {
                res[oId] = x;
                oId += 2;
            }
        }
        return res;

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

    // 915. 分割数组 (Partition Array into Disjoint Intervals)
    public int partitionDisjoint(int[] A) {
        int n = nums152.length;
        int[] rightMin = new int[n];
        rightMin[n - 1] = nums152[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            rightMin[i] = Math.min(rightMin[i + 1], nums152[i]);
        }
        int leftMax = 0;
        for (int i = 0; i < n - 1; ++i) {
            leftMax = Math.max(leftMax, nums152[i]);
            if (leftMax <= rightMin[i + 1]) {
                return i + 1;
            }
        }
        return n - 1;

    }

    // 907. 子数组的最小值之和 (Sum of Subarray Minimums) --单调栈
    public int sumSubarrayMins(int[] arr) {
        int n = arr.length;
        int res = 0;
        final int MOD = (int) (1e9 + 7);
        int[] left = new int[n];
        Arrays.fill(left, -1);
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n; ++i) {
            while (!stack.isEmpty() && arr[stack.peek()] > arr[i]) {
                stack.pop();
            }
            if (!stack.isEmpty()) {
                left[i] = stack.peek();
            }
            stack.push(i);
        }
        stack.clear();
        int[] right = new int[n];
        Arrays.fill(right, n);
        for (int i = n - 1; i >= 0; --i) {
            while (!stack.isEmpty() && arr[stack.peek()] >= arr[i]) {
                stack.pop();
            }
            if (!stack.isEmpty()) {
                right[i] = stack.peek();
            }
            stack.push(i);
        }
        for (int i = 0; i < n; ++i) {
            res = (int) ((res + (long) arr[i] * (i - left[i]) * (right[i] - i)) % MOD);
        }
        return res;

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

    // 945. 使数组唯一的最小增量 (Minimum Increment to Make Array Unique)
    public int minIncrementForUnique(int[] nums) {
        int[] counts = new int[80001];
        for (int num : nums) {
            ++counts[num];
        }
        int taken = 0;
        int res = 0;
        for (int i = 0; i < counts.length; ++i) {
            if (counts[i] > 1) {
                taken += counts[i] - 1;
                res -= (counts[i] - 1) * i;
            } else if (counts[i] == 0 && taken > 0) {
                --taken;
                res += i;
            }
        }
        return res;

    }

    // 950. 按递增顺序显示卡牌
    public int[] deckRevealedIncreasing(int[] deck) {
        if (deck == null || deck.length <= 1) {
            return deck;
        }
        Queue<Integer> queue = new LinkedList<>();
        Arrays.sort(deck);
        for (int i = deck.length - 1; i >= 0; --i) {
            queue.offer(deck[i]);
            if (i == 0) {
                break;
            }
            queue.offer(queue.poll());
        }
        int[] res = new int[deck.length];
        int index = deck.length - 1;
        while (!queue.isEmpty()) {
            res[index--] = queue.poll();
        }
        return res;
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
    public int maxWidthRamp(int[] nums) {
        int max = 0;
        for (int i = nums.length - 1; i >= max; --i) {
            for (int j = 0; j < i - max; ++j) {
                if (nums[j] <= nums[i]) {
                    max = Math.max(max, i - j);

                }
            }
        }
        return max;

    }

    // 962. 最大宽度坡
    public int maxWidthRamp2(int[] nums) {
        Integer[] B = new Integer[nums.length];
        for (int i = 0; i < B.length; ++i) {
            B[i] = i;
        }
        Arrays.sort(B, new Comparator<Integer>() {

            @Override
            public int compare(Integer o1, Integer o2) {
                return nums[o1] - nums[o2];
            }
        });
        int ans = 0;
        // B数组 遍历到当前位置的最小值（索引）
        int m = nums.length;
        for (int i : B) {
            ans = Math.max(ans, i - m);
            m = Math.min(m, i);
        }
        return ans;

    }

    // 977. 有序数组的平方 (Squares of a Sorted Array)
    public int[] sortedSquares(int[] nums) {
        int n = nums.length;
        int j = 0;
        while (j < n && nums[j] < 0) {
            ++j;
        }
        int i = j - 1;
        int[] res = new int[n];
        int index = 0;
        while (i >= 0 && j < n) {
            if (nums[i] * nums[i] < nums[j] * nums[j]) {
                res[index++] = nums[i] * nums[i];
                --i;
            } else {
                res[index++] = nums[j] * nums[j];
                ++j;
            }
        }
        while (i >= 0) {
            res[index++] = nums[i] * nums[i];
            --i;
        }
        while (j < n) {
            res[index++] = nums[j] * nums[j];
            ++j;
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

    // 1018. 可被 5 整除的二进制前缀 (Binary Prefix Divisible By 5)
    public List<Boolean> prefixesDivBy5(int[] nums) {
        List<Boolean> res = new ArrayList<>();
        int v = 0;
        for (int x : nums) {
            v = ((v << 1) + x) % 5;
            res.add(v == 0);
        }
        return res;
    }

    public int game(final int[] guess, final int[] answer) {
        int count = 0;
        for (int i = 0; i < guess.length; ++i) {
            count += guess[i] == answer[i] ? 1 : 0;
        }
        return count;

    }

    // 999. 可以被一步捕获的棋子数 (Available Captures for Rook)
    public int numRookCaptures(char[][] board) {
        int res = 0;
        int[] r = new int[2];
        int SIZE = 8;
        out: for (int i = 0; i < SIZE; ++i) {
            for (int j = 0; j < SIZE; ++j) {
                if (board[i][j] == 'R') {
                    r[0] = i;
                    r[1] = j;
                    break out;
                }
            }
        }
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        for (int[] dir : dirs) {
            int x = r[0] + dir[0];
            int y = r[1] + dir[1];
            while (x >= 0 && x < SIZE && y >= 0 && y < SIZE) {
                if (board[x][y] == 'B') {
                    break;
                } else if (board[x][y] == 'p') {
                    ++res;
                    break;
                }
                x += dir[0];
                y += dir[1];
            }
        }
        return res;

    }

    // 985. 查询后的偶数和
    public int[] sumEvenAfterQueries(int[] nums, int[][] queries) {
        int[] res = new int[queries.length];
        int sum = 0;
        for (int num : nums) {
            if (num % 2 == 0) {
                sum += num;
            }
        }
        for (int i = 0; i < queries.length; ++i) {
            int index = queries[i][1];
            int val = queries[i][0];
            if (nums[index] % 2 == 0) {
                sum -= nums[index];
            }
            nums[index] += val;
            if (nums[index] % 2 == 0) {
                sum += nums[index];
            }
            res[i] = sum;
        }
        return res;

    }

    // 1464. 数组中两元素的最大乘积 (Maximum Product of Two Elements in an Array)
    public int maxProduct1464(int[] nums) {
        int max = 0;
        int secondMax = 0;
        for (int num : nums) {
            if (num >= max) {
                secondMax = max;
                max = num;
            } else if (num >= secondMax) {
                secondMax = num;
            }
        }
        return (max - 1) * (secondMax - 1);

    }

    // 1470. 重新排列数组 (Shuffle the Array)
    public int[] shuffle(int[] nums, int n) {
        int i = 0;
        int j = n;
        int[] res = new int[n * 2];
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

    // 1014. 最佳观光组合 (Best Sightseeing Pair)
    public int maxScoreSightseeingPair(int[] values) {
        int res = Integer.MIN_VALUE;
        int mx = Integer.MIN_VALUE;
        for (int i = 0; i < values.length; ++i) {
            res = Math.max(res, values[i] - i + mx);
            mx = Math.max(mx, values[i] + i);
        }
        return res;

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

    // 1035. 不相交的线 (Uncrossed Lines)
    public int maxUncrossedLines(int[] A, int[] B) {
        int[][] dp = new int[A.length + 1][B.length + 1];
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

    // 1035. 不相交的线 (Uncrossed Lines)
    private int[] nums1_1035;
    private int[] nums2_1035;
    private int[][] memo1035;

    public int maxUncrossedLines2(int[] nums1, int[] nums2) {
        int n1 = nums1.length;
        int n2 = nums2.length;
        this.nums1_1035 = nums1;
        this.nums2_1035 = nums2;
        this.memo1035 = new int[n1][n2];
        for (int i = 0; i < n1; ++i) {
            Arrays.fill(memo1035[i], -1);
        }
        return dfs1035(n1 - 1, n2 - 1);

    }

    private int dfs1035(int i, int j) {
        if (i < 0 || j < 0) {
            return 0;
        }
        if (memo1035[i][j] != -1) {
            return memo1035[i][j];
        }
        if (nums1_1035[i] == nums2_1035[j]) {
            return memo1035[i][j] = dfs1035(i - 1, j - 1) + 1;
        }
        return memo1035[i][j] = Math.max(dfs1035(i - 1, j), dfs1035(i, j - 1));
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

    // 1053. 交换一次的先前排列 (Previous Permutation With One Swap)
    public int[] prevPermOpt1(int[] arr) {
        int n = arr.length;
        int i = n - 1;
        while (i >= 1) {
            if (arr[i - 1] > arr[i]) {
                break;
            }
            --i;
        }
        if (i == 0) {
            return arr;
        }
        --i;
        int j = n - 1;
        while (arr[j] >= arr[i] || arr[j] == arr[j - 1]) {
            --j;
        }
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
        return arr;

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

    // 1122. 数组的相对排序
    // 剑指 Offer II 075. 数组相对排序
    public int[] relativeSortArray(int[] arr1, int[] arr2) {
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
    // 剑指 Offer II 075. 数组相对排序
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

    // 1128. 等价多米诺骨牌对的数量 (Number of Equivalent Domino Pairs)
    public int numEquivDominoPairs(int[][] dominoes) {
        int[] cnts = new int[1 << 10];
        int res = 0;
        for (int[] d : dominoes) {
            int cur = (1 << d[0]) | (1 << d[1]);
            res += cnts[cur];
            ++cnts[cur];
        }
        return res;

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

    // 1365. 有多少小于当前数字的数字 (How Many Numbers Are Smaller Than the Current Number)
    public int[] smallerNumbersThanCurrent(int[] nums) {
        int[] res = new int[nums.length];
        int[] counts = new int[101];
        for (int num : nums) {
            ++counts[num];
        }
        for (int i = 1; i < counts.length; ++i) {
            counts[i] += counts[i - 1];
        }
        for (int i = 0; i < res.length; ++i) {
            if (nums[i] > 0) {
                res[i] = counts[nums[i] - 1];
            }

        }
        return res;

    }

    // 1351. 统计有序矩阵中的负数 (Count Negative Numbers in a Sorted Matrix)
    public int countNegatives(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int i = 0;
        int j = n - 1;
        int res = 0;
        while (i < m && j >= 0) {
            if (grid[i][j] < 0) {
                res += m - i;
                --j;
            } else {
                ++i;
            }
        }
        return res;

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

    // 1333. 餐厅过滤器 (Filter Restaurants by Vegan-Friendly, Price and Distance)
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

    // 1366. 通过投票对团队排名 (Rank Teams by Votes)
    public String rankTeams(String[] votes) {
        int n = votes[0].length();
        int[][] cnt = new int[26][n + 1];
        for (String v : votes) {
            for (int i = 0; i < v.length(); ++i) {
                ++cnt[v.charAt(i) - 'A'][i];
            }
        }
        for (int i = 0; i < 26; ++i) {
            cnt[i][n] = i;
        }
        Arrays.sort(cnt, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                for (int i = 0; i < n; ++i) {
                    if (o1[i] != o2[i]) {
                        return Integer.compare(o2[i], o1[i]);
                    }
                }
                return Integer.compare(o1[n], o2[n]);
            }
        });
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < n; ++i) {
            res.append((char) (cnt[i][n] + 'A'));
        }
        return res.toString();

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
    public int maxAliveYear2(int[] birth, int[] death) {
        int[] counts = new int[120];
        for (int i = 0; i < birth.length; ++i) {
            ++counts[birth[i] - 1900];
            --counts[death[i] - 1900 + 1];
        }
        int max = counts[0];
        int maxIndex = 0;
        for (int i = 1; i < counts.length; ++i) {
            counts[i] += counts[i - 1];
            if (max < counts[i]) {
                max = counts[i];
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
    public int[] masterMind(String solution, String guess) {
        int[] res = new int[2];
        // R、G、B、Y
        int[] counts = new int[4];
        for (int i = 0; i < guess.length(); ++i) {
            if (guess.charAt(i) == solution.charAt(i)) {
                ++res[0];
            }
            if (guess.charAt(i) == 'R') {
                ++counts[0];
            } else if (guess.charAt(i) == 'G') {
                ++counts[1];
            } else if (guess.charAt(i) == 'B') {
                ++counts[2];
            } else if (guess.charAt(i) == 'Y') {
                ++counts[3];
            }
        }
        for (char c : solution.toCharArray()) {
            if (c == 'R' && counts[0]-- > 0) {
                ++res[1];
            } else if (c == 'G' && counts[1]-- > 0) {
                ++res[1];
            } else if (c == 'B' && counts[2]-- > 0) {
                ++res[1];
            } else if (c == 'Y' && counts[3]-- > 0) {
                ++res[1];
            }
        }
        res[1] -= res[0];
        return res;

    }

    // 1144. 递减元素使数组呈锯齿状 (Decrease Elements To Make Array Zigzag)
    public int movesToMakeZigzag(int[] nums) {
        int n = nums.length;
        int count1 = 0;
        for (int i = 1; i < n; i += 2) {
            int min = nums[i - 1];
            if (i + 1 < n) {
                min = Math.min(min, nums[i + 1]);
            }
            if (nums[i] >= min) {
                count1 += nums[i] - min + 1;
            }
        }
        int count2 = 0;
        for (int i = 0; i < n; i += 2) {
            int min = Integer.MAX_VALUE;
            if (i + 1 < n) {
                min = nums[i + 1];
            }
            if (i - 1 >= 0) {
                min = Math.min(min, nums[i - 1]);
            }
            if (min != Integer.MAX_VALUE && nums[i] >= min) {
                count2 += nums[i] - min + 1;
            }
        }
        return Math.min(count1, count2);

    }

    // 1146. 快照数组 (Snapshot Array)
    class SnapshotArray {
        private List<int[]>[] list;

        private int id_snap;

        public SnapshotArray(int length) {
            this.list = new ArrayList[length];
            Arrays.setAll(list, k -> new ArrayList<>());
        }

        public void set(int index, int val) {
            List<int[]> items = list[index];
            if (items.isEmpty() || items.get(items.size() - 1)[1] != id_snap) {
                items.add(new int[] { val, id_snap });
            } else {
                items.set(items.size() - 1, new int[] { val, id_snap });
            }
        }

        public int snap() {
            return id_snap++;
        }

        public int get(int index, int snap_id) {
            return bin(list[index], snap_id);
        }

        private int bin(List<int[]> items, int snap_id) {
            int n = items.size();
            int left = 0;
            int right = n - 1;
            int res = 0;
            while (left <= right) {
                int mid = left + ((right - left) >> 1);
                if (items.get(mid)[1] <= snap_id) {
                    res = items.get(mid)[0];
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            return res;
        }
    }

    // 169. 多数元素 // 剑指 Offer 39. 数组中出现次数超过一半的数字 // 面试题 17.10. Find Majority Element
    // LCCI
    public int majorityElement7(final int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];

    }

    // 169. 多数元素 // 剑指 Offer 39. 数组中出现次数超过一半的数字 // 面试题 17.10. Find Majority Element
    // LCCI
    public int majorityElement6(int[] nums) {
        int count = 0;
        int majorityNum = nums[0];
        for (int num : nums) {
            if (count == 0) {
                majorityNum = num;
            }
            count += majorityNum == num ? 1 : -1;
        }
        // 面试题 17.10可能不能保证超过一半的数字一定存在，需要检查该数字是否存在
        count = 0;
        for (int num : nums) {
            if (num == majorityNum) {
                ++count;
            }
        }
        return count * 2 > nums.length ? majorityNum : -1;

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

    // 1170. 比较字符串最小字母出现频次 (Compare Strings by Frequency of the Smallest Character)
    public int[] numSmallerByFrequency(String[] queries, String[] words) {
        int[] cnts = new int[12];
        for (String word : words) {
            ++cnts[check1170(word)];
        }
        for (int i = 1; i < cnts.length; ++i) {
            cnts[i] += cnts[i - 1];
        }
        int n = queries.length;
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            int c = check1170(queries[i]);
            res[i] = cnts[11] - cnts[c];
        }
        return res;
    }

    private int check1170(String s) {
        int min = 26;
        int cnt = 0;
        for (char c : s.toCharArray()) {
            int cur = c - 'a';
            if (cur < min) {
                min = cur;
                cnt = 1;
            } else if (cur == min) {
                ++cnt;
            }
        }
        return cnt;
    }

    // 1170. 比较字符串最小字母出现频次 (Compare Strings by Frequency of the Smallest Character)
    // --线段树
    private int[] seg1170;

    public int[] numSmallerByFrequency2(String[] queries, String[] words) {
        final int N = 10;
        this.seg1170 = new int[N * 4];
        for (String word : words) {
            int cnt = check1170(word);
            insert1170(1, 1, N, cnt);
        }
        int n = queries.length;
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            int cnt = check1170(queries[i]);
            if (cnt < N) {
                res[i] = query1170(1, 1, N, cnt + 1, N);
            }
        }
        return res;
    }

    private int query1170(int o, int l, int r, int L, int R) {
        if (L <= l && r <= R) {
            return seg1170[o];
        }
        int mid = l + ((r - l) >> 1);
        int cnt = 0;
        if (L <= mid) {
            cnt += query1170(o * 2, l, mid, L, R);
        }
        if (R >= mid + 1) {
            cnt += query1170(o * 2 + 1, mid + 1, r, L, R);
        }
        return cnt;
    }

    private void insert1170(int o, int l, int r, int id) {
        if (l == r) {
            ++seg1170[o];
            return;
        }
        int mid = l + ((r - l) >> 1);
        if (id <= mid) {
            insert1170(o * 2, l, mid, id);
        } else {
            insert1170(o * 2 + 1, mid + 1, r, id);
        }
        seg1170[o] = seg1170[o * 2] + seg1170[o * 2 + 1];
    }

    // 1331. 数组序号转换 (Rank Transform of an Array)
    public int[] arrayRankTransform(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        int n = arr.length;
        int[] res = arr.clone();
        Arrays.sort(res);
        int index = 1;
        for (int num : res) {
            if (!map.containsKey(num)) {
                map.put(num, index++);
            }
        }
        for (int i = 0; i < n; ++i) {
            res[i] = map.get(arr[i]);
        }
        return res;

    }

    // 1184. 公交站间的距离 (Distance Between Bus Stops)
    public int distanceBetweenBusStops(final int[] distance, int start, int destination) {
        if (start > destination) {
            int temp = start;
            start = destination;
            destination = temp;
        }
        int res = 0;
        for (int i = start; i < destination; ++i) {
            res += distance[i];
        }
        return Math.min(res, Arrays.stream(distance).sum() - res);

    }

    // 1185. 一周中的第几天 (Day of the Week)
    public String dayOfTheWeek(int day, int month, int year) {
        int[] mon = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        List<String> list = List.of("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday");
        int res = 4;
        for (int y = 1971; y < year; ++y) {
            boolean isLeapYear = y % 4 == 0 && y % 100 != 0 || y % 400 == 0;
            res += isLeapYear ? 366 : 365;
        }
        for (int m = 1; m < month; ++m) {
            res += mon[m - 1];
            if (m == 2 && (year % 4 == 0 && year % 100 != 0 || year % 400 == 0)) {
                res += 1;
            }
        }
        res += day;
        return list.get(res % 7);

    }

    // 1200. 最小绝对差 (Minimum Absolute Difference)
    public List<List<Integer>> minimumAbsDifference(int[] arr) {
        Arrays.sort(arr);
        List<List<Integer>> res = new ArrayList<>();
        int minDiff = Integer.MAX_VALUE;
        for (int i = 1; i < arr.length; ++i) {
            if (arr[i] - arr[i - 1] < minDiff) {
                minDiff = arr[i] - arr[i - 1];
                res.clear();
                res.add(Arrays.asList(arr[i - 1], arr[i]));
            } else if (arr[i] - arr[i - 1] == minDiff) {
                res.add(Arrays.asList(arr[i - 1], arr[i]));
            }
        }
        return res;

    }

    // 1217. 玩筹码 (Minimum Cost to Move Chips to The Same Position)
    public int minCostToMoveChips(int[] position) {
        int[] cnts = new int[2];
        for (int p : position) {
            ++cnts[p & 1];
        }
        return Math.min(cnts[0], cnts[1]);
    }

    // 1222. 可以攻击国王的皇后 (Queens That Can Attack the King)
    public List<List<Integer>> queensAttacktheKing(int[][] queens, int[] king) {
        int n = 8;
        Set<Integer> set = new HashSet<>();
        for (int[] q : queens) {
            set.add(q[0] * n + q[1]);
        }
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 }, { 1, 0 }, { -1, 0 } };
        List<List<Integer>> res = new ArrayList<>();
        for (int[] d : dirs) {
            int x = king[0] + d[0];
            int y = king[1] + d[1];
            while (x < n && x >= 0 && y < n && y >= 0) {
                if (set.contains(x * n + y)) {
                    res.add(List.of(x, y));
                    break;
                }
                x += d[0];
                y += d[1];
            }
        }
        return res;

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

    // 1037. 有效的回旋镖 (判断三点是否共线)
    public boolean isBoomerang(int[][] points) {
        for (int i = 1; i < points.length; ++i) {
            points[i][0] -= points[0][0];
            points[i][1] -= points[0][1];
            // 判断三个点是否有重复
            if (points[i][0] == 0 && points[i][1] == 0) {
                return false;
            }
        }
        int A = -points[1][1];
        int B = points[1][0];
        return (A * points[2][0] + B * points[2][1]) != 0;

    }

    // 1252. 奇数值单元格的数目
    // 利用 奇数 + 偶数 = 奇数 的特性
    public int oddCells4(int m, int n, int[][] indices) {
        int[] rows = new int[m];
        int[] cols = new int[n];
        for (int[] indice : indices) {
            ++rows[indice[0]];
            ++cols[indice[1]];
        }
        int rowOdds = 0;
        int colOdds = 0;
        for (int row : rows) {
            if (row % 2 == 1) {
                ++rowOdds;
            }
        }
        for (int col : cols) {
            if (col % 2 == 1) {
                ++colOdds;
            }
        }
        return rowOdds * (n - colOdds) + colOdds * (m - rowOdds);

    }

    // 1260. 二维网格迁移 (Shift 2D Grid)
    public List<List<Integer>> shiftGrid3(int[][] grid, int k) {
        int m = grid.length;
        int n = grid[0].length;
        int[][] res = new int[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int newJ = (j + k) % n;
                int newI = (i + (j + k) / n) % m;
                res[newI][newJ] = grid[i][j];
            }
        }
        List<List<Integer>> list = new ArrayList<>();
        for (int i = 0; i < m; ++i) {
            List<Integer> subList = new ArrayList<>();
            for (int j = 0; j < n; ++j) {
                subList.add(res[i][j]);
            }
            list.add(subList);
        }
        return list;

    }

    // 1266. 访问所有点的最小时间 (Minimum Time Visiting All Points)
    public int minTimeToVisitAllPoints(final int[][] points) {
        int res = 0;
        for (int i = 1; i < points.length; ++i) {
            res += Math.max(Math.abs(points[i][1] - points[i - 1][1]), Math.abs(points[i][0] - points[i - 1][0]));
        }
        return res;

    }

    // 1275. 找出井字棋的获胜者
    public String tictactoe(int[][] moves) {
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
            if (counts[moves[i][0]] == 3 || counts[moves[i][1] + 3] == 3 || counts[6] == 3 || counts[7] == 3) {
                if (moves.length % 2 == 1) {
                    return "A";
                } else {
                    return "B";
                }
            }
        }
        if (moves.length == 9) {
            return "Draw";
        }
        return "Pending";

    }

    // 面试题 16.04. 井字游戏
    public String tictactoe(String[] board) {
        int n = board.length;
        int[][] counts = new int[2][2 * n + 2];
        int slots = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                char c = board[i].charAt(j);
                if (c == 'O') {
                    ++counts[0][i];
                    ++counts[0][j + n];
                    if (i == j) {
                        ++counts[0][counts[0].length - 2];
                    }
                    if (i + j == n - 1) {
                        ++counts[0][counts[0].length - 1];
                    }
                    ++slots;
                } else if (c == 'X') {
                    ++counts[1][i];
                    ++counts[1][j + n];
                    if (i == j) {
                        ++counts[1][counts[0].length - 2];
                    }
                    if (i + j == n - 1) {
                        ++counts[1][counts[0].length - 1];
                    }
                    ++slots;
                }
            }
        }
        for (int i = 0; i < counts.length; ++i) {
            for (int j = 0; j < counts[i].length; ++j) {
                if (counts[i][j] == n) {
                    return i == 0 ? "O" : "X";
                }
            }
        }
        return slots == n * n ? "Draw" : "Pending";

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

    // 1752. 检查数组是否经排序和轮转得到 (Check if Array Is Sorted and Rotated)
    public boolean check(int[] nums) {
        int n = nums.length;
        int i = 1;
        int count = 0;
        while (i < n) {
            if (nums[i] < nums[i - 1]) {
                if (++count > 1) {
                    return false;
                }
            }
            ++i;
        }
        return count == 0 || (count == 1 && nums[0] >= nums[n - 1]);

    }

    // 1672. 最富有客户的资产总量
    public int maximumWealth(int[][] accounts) {
        int max = 0;
        for (int[] account : accounts) {
            max = Math.max(max, Arrays.stream(account).sum());
        }
        return max;

    }

    // 1652. 拆炸弹 (Defuse the Bomb)
    public int[] decrypt(int[] code, int k) {
        int n = code.length;
        int[] res = new int[n];
        if (k == 0) {
            return res;
        }
        int[] prefix = new int[n + 1];
        for (int i = 1; i < n + 1; ++i) {
            prefix[i] = prefix[i - 1] + code[i - 1];
        }
        if (k > 0) {
            for (int i = 0; i < n; ++i) {
                res[i] = prefix[Math.min(i + k + 1, n)] - prefix[i + 1] + prefix[k - Math.min(k, (n - (i + 1)))];
            }
        } else {
            for (int i = 0; i < n; ++i) {
                res[i] = prefix[i] - prefix[Math.max(0, i + k)] + prefix[n] - prefix[n - Math.max(0, -k - i)];
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

    // 78. 子集 (Subsets)
    // LCR 079. 子集
    // 面试题 08.04. 幂集
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < 1 << n; ++i) {
            int c = i;
            List<Integer> list = new ArrayList<>();
            while (c != 0) {
                int index = Integer.numberOfTrailingZeros(c);
                list.add(nums[index]);
                c &= c - 1;
            }
            res.add(list);
        }
        return res;
    }

    // 78. 子集 (Subsets)
    // 面试题 08.04. 幂集 --回溯
    private List<List<Integer>> res78;
    private List<Integer> path78;
    private int n78;
    private int[] nums78;

    public List<List<Integer>> subsets2(int[] nums) {
        this.res78 = new ArrayList<>();
        this.path78 = new ArrayList<>();
        this.n78 = nums.length;
        this.nums78 = nums;
        dfs78(0);
        return res78;
    }

    private void dfs78(int i) {
        if (i == n78) {
            res78.add(new ArrayList<>(path78));
            return;
        }
        dfs78(i + 1);
        path78.add(nums78[i]);
        dfs78(i + 1);
        path78.remove(path78.size() - 1);
    }

    // 90. 子集 II (Subsets II) --回溯
    private int[] nums90;
    private List<List<Integer>> res90;
    private int used90;
    private int n90;
    private List<Integer> list90;

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        this.nums90 = nums;
        this.res90 = new ArrayList<>();
        this.used90 = 0;
        this.n90 = nums.length;
        this.list90 = new ArrayList<>();
        dfs90(0);
        return res90;

    }

    private void dfs90(int i) {
        if (i == n90) {
            res90.add(new ArrayList<>(list90));
            return;
        }
        dfs90(i + 1);
        if (!(i > 0 && nums90[i] == nums90[i - 1] && ((used90 >> (i - 1)) & 1) == 0)) {
            list90.add(nums90[i]);
            used90 ^= 1 << i;
            dfs90(i + 1);
            list90.remove(list90.size() - 1);
            used90 ^= 1 << i;
        }
    }

    // 1329. 将矩阵按对角线排序 (Sort the Matrix Diagonally)
    public int[][] diagonalSort(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                map.computeIfAbsent(i - j, k -> new ArrayList<>()).add(mat[i][j]);
            }
        }
        for (List<Integer> v : map.values()) {
            Collections.sort(v);
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                mat[i][j] = map.get(i - j).remove(0);
            }
        }
        return mat;

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

    // 1431. 拥有最多糖果的孩子
    public List<Boolean> kidsWithCandies(int[] candies, int extraCandies) {
        int max = Arrays.stream(candies).max().getAsInt();
        List<Boolean> res = new ArrayList<>();
        for (int candie : candies) {
            res.add(candie + extraCandies >= max);
        }
        return res;

    }

    // 1437. 是否所有 1 都至少相隔 k 个元素 (Check If All 1's Are at Least Length K Places Away)
    public boolean kLengthApart(int[] nums, int k) {
        int pre = Integer.MIN_VALUE / 2;
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] == 1) {
                if (i - pre - 1 < k) {
                    return false;
                }
                pre = i;
            }
        }
        return true;

    }

    // 1450. 在既定时间做作业的学生人数 (Number of Students Doing Homework at a Given Time)
    public int busyStudent(int[] startTime, int[] endTime, int queryTime) {
        var n = startTime.length;
        var res = 0;
        for (var i = 0; i < n; ++i) {
            if (startTime[i] <= queryTime && queryTime <= endTime[i]) {
                ++res;
            }
        }
        return res;

    }

    // 1460. 通过翻转子数组使两个数组相等 (Make Two Arrays Equal by Reversing Sub-arrays)
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

    // 1460. 通过翻转子数组使两个数组相等 (Make Two Arrays Equal by Reversing Sub-arrays)
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

    // 1640. 能否连接形成数组 (Check Array Formation Through Concatenation)
    public boolean canFormArray(int[] arr, int[][] pieces) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] piece : pieces) {
            List<Integer> list = new ArrayList<>();
            for (int num : piece) {
                list.add(num);
            }
            map.put(piece[0], list);
            // 自动装箱
            // map.put(piece[0], Arrays.stream(piece).boxed().collect(Collectors.toList()));
        }
        for (int i = 0; i < arr.length; ++i) {
            if (!map.containsKey(arr[i])) {
                return false;
            }
            List<Integer> list = map.get(arr[i]);
            for (int j = 0; j < list.size(); ++j) {
                if (i == arr.length) {
                    return false;
                }
                if (arr[i] != list.get(j)) {
                    return false;
                }
                ++i;
            }
            --i;
        }
        return true;

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
        long waitTime = 0L;
        long endTime = 0L;
        for (int[] customer : customers) {
            if (endTime > customer[0]) {
                endTime += customer[1];
                waitTime += endTime - customer[0];
            } else {
                endTime = customer[0] + customer[1];
                waitTime += customer[1];
            }
        }
        return (double) waitTime / customers.length;

    }

    // 1726. 同积元组 (Tuple with Same Product)
    public int tupleSameProduct(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                res += map.merge(nums[i] * nums[j], 1, Integer::sum) - 1;
            }
        }
        return res * 8;

    }

    // 1758. 生成交替二进制字符串的最少操作数 (Minimum Changes To Make Alternating Binary String)
    public int minOperations1758(String s) {
        int n = s.length();
        // 1 0 1 0 1 。。。
        int count1 = 0;
        // 0 1 0 1 0 。。。
        int count2 = 0;
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) - '0' == i % 2) {
                ++count1;
            } else {
                ++count2;
            }
        }
        return Math.min(count1, count2);

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

    // 1512. 好数对的数目 (Number of Good Pairs)
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

    // 1572. 矩阵对角线元素的和 (Matrix Diagonal Sum)
    public int diagonalSum(int[][] mat) {
        int n = mat.length;
        int j1 = 0;
        int j2 = n - 1;
        int res = 0;
        for (int i = 0; i < n; ++i) {
            res += mat[i][j1] + mat[i][j2];
            ++j1;
            --j2;
        }
        return n % 2 == 1 ? res - mat[n / 2][n / 2] : res;

    }

    // 1608. 特殊数组的特征值--o(n) (Special Array With X Elements Greater Than or
    // Equal X)
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

    // 1608. 特殊数组的特征值--o(log(n)) (Special Array With X Elements Greater Than or
    // Equal X)
    public int specialArray2(int[] nums) {
        int left = 0;
        int right = 0;
        for (int num : nums) {
            right = Math.max(right, num);
        }
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            int count = getMoreOrEqualsCounts(nums, mid);
            if (count == mid) {
                return mid;
            }
            if (count < mid) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return -1;

    }

    private int getMoreOrEqualsCounts(int[] nums, int findNum) {
        int count = 0;
        for (int num : nums) {
            if (num >= findNum) {
                ++count;
            }
        }
        return count;
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

    // 162. 寻找峰值--二分查找
    public int findPeakElement(int[] nums) {

        int left = 0;
        int right = nums.length - 2;

        while (left < right) {
            int mid = left + ((right - left) >>> 1);
            if (nums[mid] > nums[mid + 1]) {
                right = mid - 1;
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

    // 1619. 删除某些元素后的数组均值 (Mean of Array After Removing Some Elements)
    public double trimMean(int[] arr) {
        Arrays.sort(arr);
        int n = arr.length;
        int start = (int) ((double) n * 0.05);
        int end = n - 1 - (int) ((double) n * 0.05);
        int sum = 0;
        for (int i = start; i <= end; ++i) {
            sum += arr[i];
        }
        return (double) sum / (end - start + 1);

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

    // 1282. 用户分组 (Group the People Given the Group Size They Belong To)
    public List<List<Integer>> groupThePeople(int[] groupSizes) {
        int n = groupSizes.length;
        Map<Integer, List<Integer>> map = new HashMap<>();
        List<List<Integer>> res = new ArrayList<>();

        for (int i = 0; i < n; ++i) {
            map.computeIfAbsent(groupSizes[i], k -> new ArrayList<>()).add(i);
            if (map.get(groupSizes[i]).size() == groupSizes[i]) {
                res.add(map.get(groupSizes[i]));
                map.remove(groupSizes[i]);
            }
        }
        return res;
    }

    // 1773. 统计匹配检索规则的物品数量 (Count Items Matching a Rule)
    public int countMatches(List<List<String>> items, String ruleKey, String ruleValue) {
        Map<String, Integer> map = new HashMap<>();
        map.put("type", 0);
        map.put("color", 1);
        map.put("name", 2);

        int id = map.get(ruleKey);
        int res = 0;

        for (List<String> item : items) {
            if (item.get(id).equals(ruleValue)) {
                ++res;
            }
        }
        return res;

    }

    // 1769. 移动所有球到每个盒子所需的最小操作数 (Minimum Number of Operations to Move All Balls to
    // Each Box)
    public int[] minOperations(String boxes) {
        int n = boxes.length();
        char[] arr = boxes.toCharArray();
        int[] left = new int[n];
        int count = arr[0] - '0';
        for (int i = 1; i < n; ++i) {
            left[i] += left[i - 1] + count;
            count += arr[i] - '0';
        }
        int[] right = new int[n];
        count = arr[n - 1] - '0';
        for (int i = n - 2; i >= 0; --i) {
            right[i] += right[i + 1] + count;
            count += arr[i] - '0';
        }

        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            res[i] = left[i] + right[i];
        }
        return res;

    }

    // 1742. 盒子中小球的最大数量 (Maximum Number of Balls in a Box)
    public int countBalls(int lowLimit, int highLimit) {
        int res = 0;
        int n = String.valueOf(highLimit).length();
        for (int i = 1; i <= n * 9; ++i) {
            res = Math.max(res, cal1742(i, highLimit) - cal1742(i, lowLimit - 1));
        }
        return res;
    }

    private char[] arr1742;
    private int n1742;
    private int[][] memo1742;

    private int cal1742(int s, int x) {
        this.arr1742 = String.valueOf(x).toCharArray();
        this.n1742 = arr1742.length;
        this.memo1742 = new int[n1742][s + 1];
        for (int[] r : memo1742) {
            Arrays.fill(r, -1);
        }
        return dfs1742(0, s, true);

    }

    private int dfs1742(int i, int j, boolean isLimit) {
        if (i == n1742) {
            return j == 0 ? 1 : 0;
        }
        if (!isLimit && memo1742[i][j] != -1) {
            return memo1742[i][j];
        }
        int res = 0;
        int up = isLimit ? arr1742[i] - '0' : 9;
        for (int d = 0; d <= Math.min(j, up); ++d) {
            res += dfs1742(i + 1, j - d, isLimit && up == d);
        }
        if (!isLimit) {
            memo1742[i][j] = res;
        }
        return res;
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

    // 1504. 统计全 1 子矩形 (Count Submatrices With All Ones)
    public int numSubmat(int[][] mat) {
        int m = mat.length;
        int n = mat[0].length;
        int res = 0;
        for (int top = 0; top < m; ++top) {
            int[] a = new int[n];
            for (int bottom = top; bottom < m; ++bottom) {
                int h = bottom - top + 1;
                int last = -1;
                for (int j = 0; j < n; ++j) {
                    a[j] += mat[bottom][j];
                    if (a[j] != h) {
                        last = j;
                    } else {
                        res += j - last;
                    }
                }
            }
        }
        return res;

    }

    // 1504. 统计全 1 子矩形 (Count Submatrices With All Ones) --单调栈
    public int numSubmat2(int[][] mat) {
        int n = mat[0].length;
        int[] height = new int[n];
        int res = 0;
        for (int[] row : mat) {
            for (int j = 0; j < n; ++j) {
                if (row[j] == 0) {
                    height[j] = 0;
                } else {
                    height[j] += 1;
                }
            }
            Stack<int[]> st = new Stack<>();
            // (j, f, heights[j])
            st.add(new int[] { -1, 0, -1 });
            for (int j = 0; j < n; ++j) {
                while (st.peek()[2] >= height[j]) {
                    st.pop();
                }
                int left = st.peek()[0];
                int f = st.peek()[1];
                f += (j - left) * height[j];
                res += f;
                st.add(new int[] { j, f, height[j] });
            }
        }
        return res;

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

    // 1267. 统计参与通信的服务器 (Count Servers that Communicate)
    public int countServers(int[][] grid) {
        int res = 0;
        int[] rows = new int[grid.length];
        int[] cols = new int[grid[0].length];
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                if (grid[i][j] == 1) {
                    ++rows[i];
                    ++cols[j];
                }

            }
        }
        for (int i = 0; i < grid.length; ++i) {
            for (int j = 0; j < grid[0].length; ++j) {
                if (grid[i][j] == 1) {
                    if (rows[i] > 1 || cols[j] > 1) {
                        ++res;
                    }
                }
            }
        }
        return res;

    }

    // 面试题 17.19. 消失的两个数字
    public int[] missingTwo(int[] nums) {
        int xor = 0;
        for (int num : nums) {
            xor ^= num;
        }
        int n = nums.length;
        for (int i = 1; i <= n + 2; ++i) {
            xor ^= i;
        }
        int type1 = 0;
        int type2 = 0;
        int bitDiff = xor & (-xor);
        for (int i = 1; i <= n + 2; ++i) {
            if ((i & bitDiff) != 0) {
                type1 ^= i;
            } else {
                type2 ^= i;
            }
        }

        for (int num : nums) {
            if ((num & bitDiff) != 0) {
                type1 ^= num;
            } else {
                type2 ^= num;
            }
        }
        return new int[] { type1, type2 };

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
    public int[] createTargetArray(int[] nums, int[] index) {
        List<Integer> list = new ArrayList<>();
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

    // 1375. 二进制字符串前缀一致的次数 (Number of Times Binary String Is Prefix-Aligned)
    public int numTimesAllBlue(int[] flips) {
        int res = 0;
        int max = 0;
        for (int i = 0; i < flips.length; ++i) {
            max = Math.max(max, flips[i]);
            if (max == i + 1) {
                ++res;
            }
        }
        return res;
    }

    // 5. 最长回文子串 (Longest Palindromic Substring)
    public String longestPalindrome(String s) {
        int n = s.length();
        char[] arr = s.toCharArray();
        boolean[][] isValid = new boolean[n][n];
        int left = 0;
        int right = 0;
        for (int i = n - 1; i >= 0; --i) {
            for (int j = i; j < n; ++j) {
                if (arr[i] == arr[j] && (j - i < 2 || isValid[i + 1][j - 1])) {
                    isValid[i][j] = true;
                    if (j - i > right - left) {
                        left = i;
                        right = j;
                    }
                }
            }
        }
        return s.substring(left, right + 1);

    }

    // 5. 最长回文子串 (Longest Palindromic Substring)
    // 中心扩展算法
    public String longestPalindrome2(String s) {
        int left = 0;
        int right = 0;
        for (int i = 0; i < s.length(); ++i) {
            int len1 = calculateConsecutive(s, i, i);
            int len2 = calculateConsecutive(s, i, i + 1);
            int len = Math.max(len1, len2);
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

    // 6. Z 字形变换 (Zigzag Conversion)
    public String convert(String s, int numRows) {
        if (numRows <= 1) {
            return s;
        }
        List<StringBuilder> list = new ArrayList<>();
        for (int i = 0; i < numRows; ++i) {
            list.add(new StringBuilder());
        }
        int flag = -1;
        int index = 0;
        int rowIndex = 0;
        while (index < s.length()) {
            list.get(rowIndex).append(s.charAt(index++));
            if (rowIndex == 0 || rowIndex == numRows - 1) {
                flag = -flag;
            }
            rowIndex += flag;
        }
        StringBuilder res = new StringBuilder();
        for (StringBuilder row : list) {
            res.append(row);
        }
        return res.toString();

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

    // 43. 字符串相乘
    public String multiply(String num1, String num2) {
        if (num1.equals("0") || num2.equals("0")) {
            return "0";
        }
        int m = num1.length();
        int n = num2.length();
        int[] res = new int[m + n];
        for (int i = m - 1; i >= 0; --i) {
            int number1 = num1.charAt(i) - '0';
            for (int j = n - 1; j >= 0; --j) {
                int number2 = num2.charAt(j) - '0';
                res[i + j + 1] += number1 * number2;
            }
        }
        for (int i = res.length - 1; i >= 1; --i) {
            res[i - 1] += res[i] / 10;
            res[i] %= 10;

        }
        int index = 0;
        index = res[0] == 0 ? 1 : 0;
        StringBuilder builder = new StringBuilder();
        while (index < res.length) {
            builder.append(res[index++]);
        }
        return builder.toString();

    }

    // 49. 字母异位词分组 (Group Anagrams)
    // 剑指 Offer II 033. 变位词组
    // 面试题 10.02. 变位词组 (Group Anagrams LCCI)
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String key = String.valueOf(chars);
            map.computeIfAbsent(key, k -> new ArrayList<>()).add(str);
        }
        return new ArrayList<>(map.values());
    }

    // 49. 字母异位词分组 (Group Anagrams)
    // 剑指 Offer II 033. 变位词组
    // 面试题 10.02. 变位词组 (Group Anagrams LCCI)
    public List<List<String>> groupAnagrams2(String[] strs) {
        int[] counts = new int[26];
        Map<String, List<String>> map = new HashMap<>();
        for (String s : strs) {
            Arrays.fill(counts, 0);
            char[] chars = s.toCharArray();
            for (char c : chars) {
                ++counts[c - 'a'];
            }
            StringBuilder builder = new StringBuilder();
            for (int count : counts) {
                builder.append("#").append(count);
            }
            map.computeIfAbsent(builder.toString(), k -> new ArrayList<>()).add(s);
        }
        return new ArrayList<>(map.values());
    }

    // 58. 最后一个单词的长度 (Length of Last Word)
    public int lengthOfLastWord(String s) {
        int i = s.length() - 1;
        while (i >= 0 && s.charAt(i) == ' ') {
            --i;
        }
        int j = i;
        while (i >= 0 && s.charAt(i) != ' ') {
            --i;
        }
        return j - i;
    }

    // 67. 二进制求和 (Add Binary)
    // 剑指 Offer II 002. 二进制加法
    public String addBinary(String a, String b) {
        StringBuilder res = new StringBuilder();
        int m = a.length() - 1;
        int n = b.length() - 1;
        int carry = 0;
        while (m >= 0 || n >= 0 || carry > 0) {
            if (m >= 0) {
                carry += a.charAt(m--) - '0';
            }
            if (n >= 0) {
                carry += b.charAt(n--) - '0';
            }
            res.append(carry % 2);
            carry /= 2;
        }
        return res.reverse().toString();

    }

    // 71. 简化路径
    public String simplifyPath(final String path) {
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

    // 72. 编辑距离 (Edit Distance)
    private int[][] memo72;
    private String text1_72;
    private String text2_72;

    public int minDistance(String text1, String text2) {
        int m = text1.length();
        int n = text2.length();
        memo72 = new int[m][n];
        this.text1_72 = text1;
        this.text2_72 = text2;
        for (int i = 0; i < m; ++i) {
            Arrays.fill(memo72[i], -1);
        }
        return dfs72(m - 1, n - 1);

    }

    private int dfs72(int i, int j) {
        if (i < 0) {
            return j + 1;
        }
        if (j < 0) {
            return i + 1;
        }
        if (memo72[i][j] != -1) {
            return memo72[i][j];
        }
        int min = Integer.MAX_VALUE;
        if (text1_72.charAt(i) == text2_72.charAt(j)) {
            min = Math.min(min, dfs72(i - 1, j - 1));
        } else {
            min = Math.min(min, dfs72(i - 1, j) + 1);
            min = Math.min(min, dfs72(i, j - 1) + 1);
            min = Math.min(min, dfs72(i - 1, j - 1) + 1);
        }
        return memo72[i][j] = min;
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

    // 165. 比较版本号 (Compare Version Numbers)
    public int compareVersion(String version1, String version2) {
        String[] v1 = version1.split("\\.");
        String[] v2 = version2.split("\\.");

        int n1 = v1.length;
        int n2 = v2.length;

        int n = Math.max(n1, n2);

        for (int i = 0; i < n; i++) {
            int num1 = i < n1 ? Integer.parseInt(v1[i]) : 0;
            int num2 = i < n2 ? Integer.parseInt(v2[i]) : 0;

            if (num1 > num2) {
                return 1;
            } else if (num1 < num2) {
                return -1;
            }
        }

        return 0;

    }

    // 227. 基本计算器 II
    // 面试题 16.26. 计算器
    public int calculate(String s) {
        Stack<Integer> stack = new Stack<>();
        char sign = '+';
        int i = 0;
        while (i < s.length()) {
            if (Character.isWhitespace(s.charAt(i))) {
                ++i;
                continue;
            }
            if (Character.isDigit(s.charAt(i))) {
                int num = 0;
                while (i < s.length() && Character.isDigit(s.charAt(i))) {
                    num = num * 10 + s.charAt(i++) - '0';
                }
                --i;
                if (sign == '+') {
                    stack.push(num);
                } else if (sign == '-') {
                    stack.push(-num);
                } else {
                    stack.push(getNum(stack.pop(), num, sign));
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

    // 97. 交错字符串 (Interleaving String)
    private String s1_97;
    private String s2_97;
    private String s3_97;
    private int[][] memo97;

    public boolean isInterleave(String s1, String s2, String s3) {
        this.s1_97 = s1;
        this.s2_97 = s2;
        this.s3_97 = s3;
        int n1 = s1.length();
        int n2 = s2.length();
        int n3 = s3.length();
        this.memo97 = new int[n1][n2];
        return n1 + n2 == n3 && dfs97(n1 - 1, n2 - 1);

    }

    private boolean dfs97(int i, int j) {
        if (i < 0 || j < 0) {
            return (s1_97.substring(0, i + 1) + s2_97.substring(0, j + 1)).equals(s3_97.substring(0, i + j + 2));
        }
        if (memo97[i][j] != 0) {
            return memo97[i][j] > 0;
        }
        boolean res = s1_97.charAt(i) == s3_97.charAt(i + j + 1) && dfs97(i - 1, j)
                || s2_97.charAt(j) == s3_97.charAt(i + j + 1) && dfs97(i, j - 1);
        memo97[i][j] = res ? 1 : -1;
        return res;
    }

    // 97. 交错字符串 (Interleaving String)
    // 剑指 Offer II 096. 字符串交织
    public boolean isInterleave2(final String s1, final String s2, final String s3) {
        int n1 = s1.length();
        int n2 = s2.length();
        int n3 = s3.length();
        if (n1 + n2 != n3) {
            return false;
        }
        boolean[][] dp = new boolean[n1 + 1][n2 + 1];
        dp[0][0] = true;
        for (int i = 1; i < n1 + 1; ++i) {
            dp[i][0] = dp[i - 1][0] && s1.charAt(i - 1) == s3.charAt(i - 1);
        }
        for (int j = 1; j < n2 + 1; ++j) {
            dp[0][j] = dp[0][j - 1] && s2.charAt(j - 1) == s3.charAt(j - 1);
        }

        for (int i = 1; i < n1 + 1; ++i) {
            for (int j = 1; j < n2 + 1; ++j) {
                dp[i][j] = (dp[i - 1][j] && s1.charAt(i - 1) == s3.charAt(i + j - 1))
                        || (dp[i][j - 1] && s2.charAt(j - 1) == s3.charAt(i + j - 1));
            }
        }
        return dp[n1][n2];

    }

    // 97. 交错字符串 (Interleaving String)
    // 剑指 Offer II 096. 字符串交织
    public boolean isInterleave3(final String s1, final String s2, final String s3) {
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

    // 剑指 Offer 50. 第一个只出现一次的字符
    public char firstUniqChar(String s) {
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        for (char c : s.toCharArray()) {
            if (counts[c - 'a'] == 1) {
                return c;
            }
        }
        return ' ';
    }

    // 剑指 Offer 50. 第一个只出现一次的字符
    public char firstUniqChar2(String s) {
        // character // position
        Map<Character, Integer> map = new HashMap<>();
        Queue<Pair_Offer_30> queue = new LinkedList<>();
        for (int i = 0; i < s.length(); ++i) {
            if (!map.containsKey(s.charAt(i))) {
                map.put(s.charAt(i), i);
                queue.offer(new Pair_Offer_30(i, s.charAt(i)));
            } else {
                map.put(s.charAt(i), -1);
                while (!queue.isEmpty() && map.get(queue.peek().c) == -1) {
                    queue.poll();
                }
            }
        }
        return queue.isEmpty() ? ' ' : queue.peek().c;

    }

    public class Pair_Offer_30 {
        public int pos;
        public char c;

        public Pair_Offer_30(int pos, char c) {
            this.pos = pos;
            this.c = c;
        }

    }

    // 415. 字符串相加 (Add Strings)
    public String addStrings(String num1, String num2) {
        StringBuilder res = new StringBuilder();
        int i = num1.length() - 1;
        int j = num2.length() - 1;
        int carry = 0;
        while (i >= 0 || j >= 0 || carry > 0) {
            if (i >= 0) {
                carry += num1.charAt(i--) - '0';
            }
            if (j >= 0) {
                carry += num2.charAt(j--) - '0';
            }
            res.insert(0, "" + carry % 10);
            carry /= 10;
        }
        return res.toString();

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

    public String validIPAddress(final String IP) {
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

    // 521. 最长特殊序列 Ⅰ (Longest Uncommon Subsequence I)
    public int findLUSlength(final String a, final String b) {
        return a.equals(b) ? -1 : Math.max(a.length(), b.length());
    }

    // 537. 复数乘法 (Complex Number Multiplication)
    public String complexNumberMultiply(String num1, String num2) {
        int signIndex1 = num1.indexOf("+");
        int real1 = Integer.parseInt(num1.substring(0, signIndex1));
        int imaginary1 = Integer.parseInt(num1.substring(signIndex1 + 1, num1.length() - 1));

        int signIndex2 = num2.indexOf("+");
        int real2 = Integer.parseInt(num2.substring(0, signIndex2));
        int imaginary2 = Integer.parseInt(num2.substring(signIndex2 + 1, num2.length() - 1));

        int real = real1 * real2 - imaginary1 * imaginary2;
        int imaginary = real1 * imaginary2 + real2 * imaginary1;
        return real + "+" + imaginary + "i";

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

    // 剑指 Offer II 035. 最小时间差
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

    // 1394. 找出数组中的幸运数 (Find Lucky Integer in an Array)
    public int findLucky(int[] arr) {
        int max = 0;
        for (int num : arr) {
            max = Math.max(max, num);
        }
        int[] counts = new int[max + 1];
        for (int num : arr) {
            ++counts[num];
        }
        for (int i = counts.length; i > 0; --i) {
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

    // 670. 最大交换 (Maximum Swap)
    public int maximumSwap(int num) {
        int[] last = new int[10];
        char[] chars = String.valueOf(num).toCharArray();
        int n = chars.length;
        for (int i = 0; i < n; ++i) {
            int bit = chars[i] - '0';
            last[bit] = i;
        }
        for (int i = 0; i < n; ++i) {
            int bit = chars[i] - '0';
            for (int j = 9; j > bit; --j) {
                if (last[j] > i) {
                    swap670(chars, i, last[j]);
                    return Integer.parseInt(String.valueOf(chars));
                }
            }
        }
        return num;

    }

    private void swap670(char[] chars, int i, int j) {
        char temp = chars[i];
        chars[i] = chars[j];
        chars[j] = temp;
    }

    // 900. RLE 迭代器 (RLE Iterator)
    class RLEIterator {
        private int[] A;
        private int index;

        public RLEIterator(int[] A) {
            this.A = A;
        }

        public int next(int n) {
            while (index < A.length && A[index] < n) {
                n -= A[index];
                index += 2;
            }
            if (index >= A.length) {
                return -1;
            }
            A[index] -= n;
            return A[index + 1];
        }

    }

    // 1475. 商品折扣后的最终价格 (Final Prices With a Special Discount in a Shop) --暴力
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

    // 1475. 商品折扣后的最终价格 (Final Prices With a Special Discount in a Shop) --单调栈
    public int[] finalPrices2(int[] prices) {
        Stack<Integer> stack = new Stack<>();
        int n = prices.length;
        for (int i = n - 1; i >= 0; --i) {
            int temp = prices[i];
            while (!stack.isEmpty() && prices[i] < stack.peek()) {
                stack.pop();
            }
            if (!stack.isEmpty()) {
                prices[i] -= stack.peek();
            }
            stack.push(temp);
        }
        return prices;

    }

    // 1636. 按照频率将数组升序排序 (Sort Array by Increasing Frequency)
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

    // 1636. 按照频率将数组升序排序 (Sort Array by Increasing Frequency)
    public int[] frequencySort2(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        int[][] arr = new int[map.size()][2];
        int index = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            arr[index++] = new int[] { entry.getKey(), entry.getValue() };
        }
        Arrays.sort(arr, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                if (o1[1] == o2[1]) {
                    return o2[0] - o1[0];
                }
                return o1[1] - o2[1];
            }

        });

        index = 0;
        for (int[] cur : arr) {
            for (int i = 0; i < cur[1]; ++i) {
                nums[index++] = cur[0];
            }
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
        for (int i = 1; i <= m; ++i) {
            list.add(i);
        }
        int[] res = new int[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            int index = queries[i];
            int j = 0;
            for (; j < list.size(); ++j) {
                if (list.get(j) == index) {
                    break;
                }
            }
            int val = list.remove(j);
            list.add(0, val);
            res[i] = j;
        }
        return res;

    }

    // 面试题 16.24. 数对和 (Pairs With Sum LCCI)
    public List<List<Integer>> pairSums(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int sum = nums[left] + nums[right];
            if (sum == target) {
                res.add(Arrays.asList(nums[left], nums[right]));
                ++left;
                --right;
            } else if (sum < target) {
                ++left;
            } else {
                --right;
            }
        }
        return res;

    }

    // 198. 打家劫舍 (House Robber) --记忆化搜索
    // 剑指 Offer II 089. 房屋偷盗
    private int[] nums198;
    private int[] memo198;

    public int rob(int[] nums) {
        this.nums198 = nums;
        int n = nums.length;
        memo198 = new int[n];
        Arrays.fill(memo198, -1);
        return dfs198(n - 1);

    }

    private int dfs198(int index) {
        if (index < 0) {
            return 0;
        }
        if (memo198[index] != -1) {
            return memo198[index];
        }
        int max = Math.max(dfs198(index - 2) + nums198[index], dfs198(index - 1));
        return memo198[index] = max;
    }

    // 198. 打家劫舍 (House Robber) --动态规划
    // 剑指 Offer II 089. 房屋偷盗
    public int rob198_2(int[] nums) {
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

    // 213. 打家劫舍 II (House Robber II) --记忆化搜索
    private int[] memo213;
    private int[] nums213;

    public int robII(int[] nums) {
        int n = nums.length;
        this.nums213 = nums;
        if (n <= 1) {
            return nums[0];
        }
        memo213 = new int[n];
        Arrays.fill(memo213, -1);
        int res1 = dfs213(n - 2);
        int i = 0;
        int j = n - 1;
        while (i < j) {
            int t = nums[i];
            nums[i] = nums[j];
            nums[j] = t;
            ++i;
            --j;
        }
        Arrays.fill(memo213, -1);
        int res2 = dfs213(n - 2);
        return Math.max(res1, res2);

    }

    private int dfs213(int i) {
        if (i < 0) {
            return 0;
        }
        if (memo213[i] != -1) {
            return memo213[i];
        }
        return memo213[i] = Math.max(dfs213(i - 2) + nums213[i], dfs213(i - 1));
    }

    // 213. 打家劫舍 II --动态规划
    // 剑指 Offer II 090. 环形房屋偷盗
    public int rob213II_2(int[] nums) {
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

    // 面试题 17.16. 按摩师 (类似打家劫舍)
    public int massage(int[] nums) {
        if (nums == null || nums.length == 0) {
            return 0;
        }
        if (nums.length == 1) {
            return nums[0];
        }
        int[] dp = new int[nums.length];
        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; ++i) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        return dp[dp.length - 1];
    }

    // 面试题 17.16. 按摩师 (The Masseuse LCCI)
    private int[] memo17_16;
    private int n17_16;
    private int[] nums17_16;

    public int massage2(int[] nums) {
        this.n17_16 = nums.length;
        this.memo17_16 = new int[n17_16];
        Arrays.fill(memo17_16, -1);
        this.nums17_16 = nums;
        return dfs17_16(0);

    }

    private int dfs17_16(int i) {
        if (i >= n17_16) {
            return 0;
        }
        if (memo17_16[i] != -1) {
            return memo17_16[i];
        }
        return memo17_16[i] = Math.max(dfs17_16(i + 1), dfs17_16(i + 2) + nums17_16[i]);
    }

    // 1689. 十-二进制数的最少数目
    public int minPartitions(String n) {
        int max = 0;
        for (char a : n.toCharArray()) {
            max = Math.max(max, a - '0');
        }
        return max;

    }

    // 1656. 设计有序流 (Design an Ordered Stream)
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

    // 1481. 不同整数的最少数目 (Least Number of Unique Integers after K Removals)
    public int findLeastNumOfUniqueInts(int[] arr, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : arr) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        List<Map.Entry<Integer, Integer>> list = new ArrayList<>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<Integer, Integer>>() {

            @Override
            public int compare(Map.Entry<Integer, Integer> o1, Map.Entry<Integer, Integer> o2) {
                return o1.getValue() - o2.getValue();
            }
        });
        for (int i = 0; i < list.size(); ++i) {
            Map.Entry<Integer, Integer> entry = list.get(i);
            if (k < entry.getValue()) {
                return list.size() - i;
            }
            k -= entry.getValue();
        }
        return 0;

    }

    // 剑指 Offer 62. 圆圈中最后剩下的数字
    public int lastRemaining(int n, int m) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < n; ++i) {
            list.add(i);
        }
        int index = 0;
        while (list.size() != 1) {
            index = (index + m - 1) % list.size();
            list.remove(index);
        }
        return list.get(0);

    }

    // 1007. 行相等的最少多米诺旋转 (Minimum Domino Rotations For Equal Row)
    public int minDominoRotations(int[] tops, int[] bottoms) {
        int res = Math.min(check1007(tops, bottoms, tops[0]), check1007(tops, bottoms, bottoms[0]));
        return res == Integer.MAX_VALUE ? -1 : res;
    }

    private int check1007(int[] tops, int[] bottoms, int target) {
        int toTop = 0;
        int toBottom = 0;
        for (int i = 0; i < tops.length; ++i) {
            if (tops[i] != target && bottoms[i] != target) {
                return Integer.MAX_VALUE;
            }
            if (tops[i] != target) {
                ++toTop;
            } else if (bottoms[i] != target) {
                ++toBottom;
            }
        }
        return Math.min(toTop, toBottom);
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
        List<Integer> res = new ArrayList<>();
        int start = rounds[0];
        int end = rounds[rounds.length - 1];
        if (end >= start) {
            for (int i = start; i <= end; ++i) {
                res.add(i);
            }
        } else {
            for (int i = 1; i <= end; ++i) {
                res.add(i);
            }
            for (int i = start; i <= n; ++i) {
                res.add(i);
            }
        }
        return res;

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

    // 1509. 三次操作后最大值与最小值的最小差
    public int minDifference2(int[] nums) {
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

    // 1535. 找出数组游戏的赢家 (Find the Winner of an Array Game)
    public int getWinner(int[] arr, int k) {
        int n = arr.length;
        int times = 0;
        int res = arr[0];
        int mx = arr[0];
        for (int i = 1; i < n; ++i) {
            mx = Math.max(mx, arr[i]);
            if (arr[i] > res) {
                res = arr[i];
                times = 1;
            } else {
                ++times;
            }
            if (times == k) {
                return res;
            }
        }
        return mx;
    }

    // 1827. 最少操作使数组递增 (Minimum Operations to Make the Array Increasing)
    public int minOperations(int[] nums) {
        int res = 0;
        for (int i = 1; i < nums.length; ++i) {
            if (nums[i] <= nums[i - 1]) {
                res += nums[i - 1] - nums[i] + 1;
                nums[i] = nums[i - 1] + 1;
            }
        }
        return res;

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

    // 1497. 检查数组对是否可以被 k 整除 (Check If Array Pairs Are Divisible by k)
    public boolean canArrange(int[] arr, int k) {
        int[] counts = new int[k];
        for (int num : arr) {
            ++counts[(num % k + k) % k];
        }
        for (int i = 1; i < counts.length; ++i) {
            if (counts[i] != counts[k - i]) {
                return false;
            }
        }
        return counts[0] % 2 == 0;

    }

    // 1497. 检查数组对是否可以被 k 整除 (Check If Array Pairs Are Divisible by k)
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

    // 1233. 删除子文件夹 (Remove Sub-Folders from the Filesystem)
    public List<String> removeSubfolders(String[] folder) {
        Arrays.sort(folder);
        List<String> res = new ArrayList<>();
        res.add(folder[0]);
        for (int i = 1; i < folder.length; ++i) {
            String pre = res.get(res.size() - 1);
            if (!folder[i].startsWith(pre + "/")) {
                res.add(folder[i]);
            }
        }
        return res;

    }

    // 1233. 删除子文件夹 (Remove Sub-Folders from the Filesystem)
    public List<String> removeSubfolders2(String[] folder) {
        Trie1233 trie = new Trie1233();
        for (int i = 0; i < folder.length; ++i) {
            trie.insert(folder[i], i);
        }
        List<String> res = new ArrayList<>();
        dfs1233(res, folder, trie);
        return res;

    }

    private void dfs1233(List<String> res, String[] folder, Trie1233 trie) {
        if (trie.pos != -1) {
            res.add(folder[trie.pos]);
            return;
        }
        for (Trie1233 node : trie.children.values()) {
            dfs1233(res, folder, node);
        }
    }

    public class Trie1233 {
        Map<String, Trie1233> children;
        int pos;

        public Trie1233() {
            this.children = new HashMap<>();
            this.pos = -1;
        }

        public void insert(String s, int p) {
            Trie1233 node = this;
            String[] list = s.split("/+");
            for (String sub : list) {
                node.children.putIfAbsent(sub, new Trie1233());
                node = node.children.get(sub);
            }
            node.pos = p;
        }
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
    // 剑指 Offer II 073. 狒狒吃香蕉
    public int minEatingSpeed(int[] piles, int h) {
        int res = Integer.MAX_VALUE;
        int left = 1;
        int right = Arrays.stream(piles).max().getAsInt();
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

    // 1424.对角线遍历 II (Diagonal Traverse II)
    public int[] findDiagonalOrder(List<List<Integer>> nums) {
        int count = 0;
        Map<Integer, List<Integer>> map = new TreeMap<>();
        for (int i = 0; i < nums.size(); ++i) {
            for (int j = 0; j < nums.get(i).size(); ++j) {
                map.computeIfAbsent(i + j, k -> new ArrayList<>()).add(nums.get(i).get(j));
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

    // 1814. 统计一个数组中好对子的数目 (Count Nice Pairs in an Array)
    public int countNicePairs(int[] nums) {
        final int MOD = (int) (1e9 + 7);
        long res = 0l;
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums) {
            int diff = num - rev(num);
            res = (res + map.getOrDefault(diff, 0)) % MOD;
            map.put(diff, map.getOrDefault(diff, 0) + 1);
        }
        return (int) (res % MOD);

    }

    private int rev(int num) {
        int res = 0;
        while (num != 0) {
            res = res * 10 + num % 10;
            num /= 10;
        }
        return res;
    }

    // 1711. 大餐计数
    public int countPairs(int[] deliciousness) {
        final int MOD = 1000000007;
        int res = 0;
        Map<Integer, Integer> map = new HashMap<>();
        for (int delicious : deliciousness) {
            for (int i = 0; i <= 21; ++i) {
                int remain = (1 << i) - delicious;
                if (remain >= 0) {
                    res = (res + map.getOrDefault(remain, 0)) % MOD;
                }
            }
            map.put(delicious, map.getOrDefault(delicious, 0) + 1);
        }
        return res % MOD;

    }

    // 1764. 通过连接另一个数组的子数组得到一个数组 (Form Array by Concatenating Subarrays of Another
    // Array)
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

    // 1764. 通过连接另一个数组的子数组得到一个数组 (Form Array by Concatenating Subarrays of Another
    // Array)
    public boolean canChoose2(int[][] groups, int[] nums) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int i = 0; i < nums.length; ++i) {
            map.computeIfAbsent(nums[i], k -> new ArrayList<>()).add(i);
        }
        int index = 0;
        search: for (int[] group : groups) {
            if (group.length == 0) {
                continue;
            }
            int num = group[0];
            List<Integer> list = map.getOrDefault(num, new ArrayList<>());
            int i = binarySearch1746(list, index);
            if (i == -1) {
                return false;
            }
            while (i < list.size()) {
                if (check1764(list.get(i), nums, group)) {
                    index = list.get(i) + group.length;
                    continue search;
                }
                ++i;
            }
            return false;
        }
        return true;

    }

    private boolean check1764(int i, int[] nums, int[] group) {
        int j = 0;
        while (i < nums.length && j < group.length) {
            if (nums[i] != group[j]) {
                return false;
            }
            ++i;
            ++j;
        }
        return j == group.length;
    }

    private int binarySearch1746(List<Integer> list, int target) {
        if (list.isEmpty()) {
            return -1;
        }
        int n = list.size();
        if (target > list.get(n - 1)) {
            return -1;
        }
        if (target <= list.get(0)) {
            return 0;
        }
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

    // 1733. 需要教语言的最少人数 (Minimum Number of People to Teach)
    public int minimumTeachings(int n, int[][] languages, int[][] friendships) {
        Set<Integer> s = new HashSet<>();
        for (int[] f : friendships) {
            if (!canTalk1733(f[0] - 1, f[1] - 1, languages)) {
                s.add(f[0] - 1);
                s.add(f[1] - 1);
            }
        }
        if (s.isEmpty()) {
            return 0;
        }
        Map<Integer, Integer> map = new HashMap<>();
        for (int p : s) {
            for (int l : languages[p]) {
                map.merge(l, 1, Integer::sum);
            }
        }
        return s.size() - Collections.max(map.values());
    }

    private boolean canTalk1733(int i, int j, int[][] languages) {
        for (int x : languages[i]) {
            for (int y : languages[j]) {
                if (x == y) {
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

    // 1574. 删除最短的子数组使剩余数组有序 (Shortest Subarray to be Removed to Make Array Sorted)
    public int findLengthOfShortestSubarray(int[] arr) {
        int n = arr.length;
        int right = n - 1;
        while (right > 0 && arr[right - 1] <= arr[right]) {
            --right;
        }
        if (right == 0) {
            return 0;
        }
        int res = right;
        for (int left = 0; left == 0 || arr[left - 1] <= arr[left]; ++left) {
            while (right < n && arr[left] > arr[right]) {
                ++right;
            }
            res = Math.min(res, right - left - 1);
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

    public boolean circularArrayLoop(int[] nums) {
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

    // 1800. 最大升序子数组和 (Maximum Ascending Subarray Sum)
    public int maxAscendingSum(int[] nums) {
        int res = 0;
        int cur = 0;
        for (int i = 0; i < nums.length; ++i) {
            if (i == 0 || nums[i] <= nums[i - 1]) {
                cur = nums[i];
            } else {
                cur += nums[i];
            }
            res = Math.max(res, cur);
        }
        return res;

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

    // 744. 寻找比目标字母大的最小字母 (Find Smallest Letter Greater Than Target)
    public char nextGreatestLetter(char[] letters, char target) {
        int left = 0;
        int right = letters.length - 1;
        while (left <= right) {
            int mid = left + ((right - left) >> 1);
            if (letters[mid] > target) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return letters[(right + 1) % letters.length];

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

    // 778. 水位上升的泳池中游泳 (Swim in Rising Water) --并查集
    public int swimInWater(int[][] grid) {
        int n = grid.length;
        Map<Integer, int[]> map = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                map.put(grid[i][j], new int[] { i, j });
            }
        }
        int[][] directions = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 } };
        Union778 union = new Union778(n * n);
        for (int i = 0; i < n * n; ++i) {
            int[] cur = map.get(i);
            int x = cur[0];
            int y = cur[1];
            for (int[] d : directions) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n && grid[nx][ny] <= i) {
                    union.union(x * n + y, nx * n + ny);
                }
            }
            if (union.isConnected(0, n * n - 1)) {
                return i;
            }
        }
        return -1;

    }

    public class Union778 {
        private int[] rank;
        private int[] parent;

        public Union778(int n) {
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

    // 778. 水位上升的泳池中游泳 (Swim in Rising Water) --Dijkstra
    public int swimInWater2(int[][] grid) {
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int n = grid.length;
        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[2], o2[2]);
            }

        });

        queue.offer(new int[] { 0, 0, grid[0][0] });
        int res = 0;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            int h = cur[2];
            res = Math.max(res, h);
            if (x == n - 1 && y == n - 1) {
                return res;
            }
            grid[x][y] = -1;
            for (int[] d : dirs) {
                int nx = x + d[0];
                int ny = y + d[1];
                if (nx >= 0 && nx < n && ny >= 0 && ny < n && grid[nx][ny] != -1) {
                    queue.offer(new int[] { nx, ny, grid[nx][ny] });
                }
            }
        }
        return -1;

    }

    // 1319. 连通网络的操作次数 (Number of Operations to Make Network Connected) --并查集
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

    // 684. 冗余连接 (Redundant Connection) --并查集
    // 剑指 Offer II 118. 多余的边
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

    // 547. 省份数量 (Number of Provinces) --并查集
    // 剑指 Offer II 119. 最长连续序列
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

    // 547. 省份数量 (Number of Provinces) --bfs
    // 剑指 Offer II 119. 最长连续序列
    public int findCircleNum2(int[][] isConnected) {
        int n = isConnected.length;
        boolean[] visited = new boolean[n];
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                Queue<Integer> queue = new LinkedList<>();
                queue.offer(i);
                while (!queue.isEmpty()) {
                    int cur = queue.poll();
                    visited[cur] = true;
                    for (int j = i + 1; j < n; ++j) {
                        if (!visited[j] && isConnected[cur][j] == 1) {
                            visited[j] = true;
                            queue.offer(j);
                        }
                    }
                }
                ++res;
            }
        }
        return res;

    }

    // 959. 由斜杠划分区域 (Regions Cut By Slashes) --bfs
    public int regionsBySlashes(String[] grid) {
        int n = grid.length;
        // 每个格，转换为 3 * 3 矩阵 相应的斜线置为1
        int[][] arr = new int[n * 3][n * 3];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                char c = grid[i].charAt(j);
                if (c == '/') {
                    arr[i * 3][j * 3 + 2] = 1;
                    arr[i * 3 + 1][j * 3 + 1] = 1;
                    arr[i * 3 + 2][j * 3] = 1;
                } else if (c == '\\') {
                    arr[i * 3][j * 3] = 1;
                    arr[i * 3 + 1][j * 3 + 1] = 1;
                    arr[i * 3 + 2][j * 3 + 2] = 1;
                }
            }
        }
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int m = arr.length;
        Queue<int[]> queue = new LinkedList<>();
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < m; ++j) {
                if (arr[i][j] == 0) {
                    ++res;
                    queue.offer(new int[] { i, j });
                    arr[i][j] = 1;
                    while (!queue.isEmpty()) {
                        int[] cur = queue.poll();
                        for (int[] direction : directions) {
                            int nx = cur[0] + direction[0];
                            int ny = cur[1] + direction[1];
                            if (nx >= 0 && nx < m && ny >= 0 && ny < m && arr[nx][ny] == 0) {
                                arr[nx][ny] = 1;
                                queue.offer(new int[] { nx, ny });
                            }
                        }
                    }
                }
            }
        }
        return res;
    }

    // 959. 由斜杠划分区域 (Regions Cut By Slashes) --并查集
    public int regionsBySlashes2(String[] grid) {
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

    // 839. 相似字符串组 (Similar String Groups) --bfs
    // 剑指 Offer II 117. 相似的字符串
    public int numSimilarGroups(String[] strs) {
        int res = 0;
        int n = strs.length;
        boolean[] visited = new boolean[n];
        Queue<String> queue = new LinkedList<>();
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                ++res;
                visited[i] = true;
                queue.offer(strs[i]);
                while (!queue.isEmpty()) {
                    String cur = queue.poll();
                    for (int j = i + 1; j < n; ++j) {
                        if (!visited[j] && similar839(cur, strs[j])) {
                            visited[j] = true;
                            queue.offer(strs[j]);
                        }
                    }
                }
            }
        }
        return res;

    }

    private boolean similar839(String s1, String s2) {
        int diff = 0;
        for (int i = 0; i < s1.length(); ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                if (++diff > 2) {
                    return false;
                }
            }
        }
        return diff == 0 || diff == 2;
    }

    // 839. 相似字符串组 (Similar String Groups) --并查集
    // 剑指 Offer II 117. 相似的字符串
    public int numSimilarGroups2(String[] strs) {
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
        return count == 0 || count == 2;
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

    // 1631. 最小体力消耗路径 (Path With Minimum Effort) --bfs
    public int minimumEffortPath(int[][] heights) {
        int left = 0;
        int right = 1000000;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check1631(heights, mid)) {
                res = mid;
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return res;
    }

    private boolean check1631(int[][] heights, int h) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };
        int m = heights.length;
        int n = heights[0].length;
        boolean[][] visited = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { 0, 0 });
        visited[0][0] = true;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            if (x == m - 1 && y == n - 1) {
                return true;
            }
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]
                        && Math.abs(heights[x][y] - heights[nx][ny]) <= h) {
                    queue.offer(new int[] { nx, ny });
                    visited[nx][ny] = true;
                }
            }
        }
        return false;
    }

    // 1631. 最小体力消耗路径 (Path With Minimum Effort) --并查集
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

    // 1631. 最小体力消耗路径 (Path With Minimum Effort) --Dijkstra
    public int minimumEffortPath3(int[][] heights) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };
        int m = heights.length;
        int n = heights[0].length;
        boolean[][] visited = new boolean[m][n];
        int[][] distances = new int[m][n];
        for (int i = 0; i < m; ++i) {
            Arrays.fill(distances[i], Integer.MAX_VALUE);
        }
        Queue<int[]> queue = new PriorityQueue<>(new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[2] - o2[2];
            }

        });
        queue.offer(new int[] { 0, 0, 0 });
        distances[0][0] = 0;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            int d = cur[2];
            if (visited[x][y]) {
                continue;
            }
            visited[x][y] = true;
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    if (Math.max(Math.abs(heights[x][y] - heights[nx][ny]), d) < distances[nx][ny]) {
                        distances[nx][ny] = Math.max(Math.abs(heights[x][y] - heights[nx][ny]), d);
                        queue.offer(new int[] { nx, ny, distances[nx][ny] });
                    }
                }
            }
        }
        return distances[m - 1][n - 1];

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

    // 130. 被围绕的区域 (Surrounded Regions) --并查集
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

    // 130. 被围绕的区域 (Surrounded Regions) --bfs
    public void solvw2(char[][] board) {
        int m = board.length;
        int n = board[0].length;
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 'O') {
                    if (i == 0 || j == 0 || i == m - 1 || j == n - 1) {
                        queue.offer(new int[] { i, j });
                        board[i][j] = 'A';
                    }
                }
            }
        }
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                    if (board[nx][ny] == 'O') {
                        board[nx][ny] = 'A';
                        queue.offer(new int[] { nx, ny });
                    }
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }
    }

    // 130. 被围绕的区域 (Surrounded Regions) --dfs
    public void solve3(char[][] board) {
        int m = board.length;
        int n = board[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if ((i == 0 || i == m - 1 || j == 0 || j == n - 1) && board[i][j] == 'O') {
                    dfs130(board, i, j);
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 'A') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }

    }

    private void dfs130(char[][] board, int i, int j) {
        int m = board.length;
        int n = board[0].length;
        if (!(i >= 0 && i < m && j >= 0 && j < n && board[i][j] == 'O')) {
            return;
        }
        board[i][j] = 'A';
        dfs130(board, i - 1, j);
        dfs130(board, i + 1, j);
        dfs130(board, i, j - 1);
        dfs130(board, i, j + 1);
    }

    // 721. 账户合并 (Accounts Merge) --并查集
    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        // name <- post <-> id
        int id = 0;
        Map<String, Integer> postToId = new HashMap<>();
        Map<Integer, String> idToPost = new HashMap<>();
        Map<String, String> postToName = new HashMap<>();
        for (List<String> account : accounts) {
            String name = account.get(0);
            for (int i = 1; i < account.size(); ++i) {
                if (!postToId.containsKey(account.get(i))) {
                    postToId.put(account.get(i), id);
                    idToPost.put(id, account.get(i));
                    ++id;
                }
                postToName.put(account.get(i), name);
            }
        }
        Union721 union = new Union721(id);
        for (List<String> account : accounts) {
            for (int i = 2; i < account.size(); ++i) {
                union.union(postToId.get(account.get(1)), postToId.get(account.get(i)));
            }
        }
        Map<Integer, List<String>> rootToList = new HashMap<>();
        for (int i = 0; i < id; ++i) {
            int root = union.getRoot(i);
            rootToList.computeIfAbsent(root, k -> new ArrayList<>()).add(idToPost.get(i));
        }
        List<List<String>> res = new ArrayList<>();
        for (List<String> vals : rootToList.values()) {
            List<String> subRes = new ArrayList<>();
            String name = postToName.get(vals.get(0));
            Collections.sort(vals);
            subRes.add(name);
            subRes.addAll(vals);
            res.add(subRes);
        }
        return res;
    }

    class Union721 {
        private int[] parent;
        private int[] rank;

        public Union721(int n) {
            this.parent = new int[n];
            this.rank = new int[n];
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
                parent[root2] = root1;
            } else {
                parent[root1] = root2;
                if (rank[root1] == rank[root2]) {
                    ++rank[root2];
                }
            }
        }

    }

    // 721. 账户合并 (Accounts Merge) --bfs
    public List<List<String>> accountsMerge2(List<List<String>> accounts) {
        Map<String, Set<String>> graph = new HashMap<>();
        for (List<String> account : accounts) {
            for (int i = 2; i < account.size(); ++i) {
                graph.computeIfAbsent(account.get(i), k -> new HashSet<>()).add(account.get(i - 1));
                graph.computeIfAbsent(account.get(i - 1), k -> new HashSet<>()).add(account.get(i));

            }
        }
        List<List<String>> res = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        for (List<String> account : accounts) {
            Queue<String> queue = new LinkedList<>();
            List<String> sub = new ArrayList<>();
            for (int i = 1; i < account.size(); ++i) {
                if (!visited.contains(account.get(i))) {
                    visited.add(account.get(i));
                    queue.offer(account.get(i));
                    sub.add(account.get(i));
                }
            }
            if (sub.isEmpty()) {
                continue;
            }
            while (!queue.isEmpty()) {
                String cur = queue.poll();
                for (String neighbor : graph.getOrDefault(cur, new HashSet<>())) {
                    if (!visited.contains(neighbor)) {
                        visited.add(neighbor);
                        queue.offer(neighbor);
                        sub.add(neighbor);
                    }
                }
            }
            Collections.sort(sub);
            sub.add(0, account.get(0));
            res.add(sub);
        }
        return res;
    }

    // 1869. 哪种连续子字符串更长
    public boolean checkZeroOnes(String s) {
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

    // 739. 每日温度 // 剑指 Offer II 038. 每日温度
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

    // 856. 括号的分数 (Score of Parentheses)
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

    // 856. 括号的分数 (Score of Parentheses)
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

    // 856. 括号的分数 (Score of Parentheses) --分治
    public int scoreOfParentheses3(String s) {
        int n = s.length();
        if (n == 2) {
            return 1;
        }
        int len = 0;
        int bal = 0;
        for (int i = 0; i < n; ++i) {
            if (s.charAt(i) == '(') {
                ++bal;
            } else {
                --bal;
            }
            if (bal == 0) {
                len = i + 1;
                break;
            }
        }
        if (len == n) {
            return 2 * scoreOfParentheses(s.substring(1, len - 1));
        } else {
            return scoreOfParentheses(s.substring(0, len)) + scoreOfParentheses(s.substring(len));
        }

    }

    // 901. 股票价格跨度 (Online Stock Span) --单调栈
    class StockSpanner {
        private Stack<int[]> stack;
        private int i;

        public StockSpanner() {
            this.stack = new Stack<>();
            stack.push(new int[] { -1, 0 });
        }

        public int next(int price) {
            while (stack.size() > 1 && stack.peek()[1] <= price) {
                stack.pop();
            }
            int res = i - stack.peek()[0];
            stack.push(new int[] { i++, price });
            return res;
        }
    }

    // 921. 使括号有效的最少添加 (Minimum Add to Make Parentheses Valid)
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

    // 921. 使括号有效的最少添加 (Minimum Add to Make Parentheses Valid)
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
    private int ptr;

    public String decodeString(String s) {
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

    private String getString(List<String> sub) {
        StringBuilder builder = new StringBuilder();
        for (String s : sub) {
            builder.append(s);
        }
        return builder.toString();
    }

    private String getNum(String s) {
        StringBuilder builder = new StringBuilder();
        while (Character.isDigit(s.charAt(ptr))) {
            builder.append(s.charAt(ptr));
            ++ptr;
        }
        return builder.toString();
    }

    // 1441. 用栈操作构建数组 (Build an Array With Stack Operations)
    public List<String> buildArray(int[] target, int n) {
        List<String> res = new ArrayList<>();
        int index = 0;
        int num = 1;
        while (index < target.length) {
            res.add("Push");
            if (target[index] == num++) {
                ++index;
            } else {
                res.add("Pop");
            }
        }
        return res;

    }

    // 682. 棒球比赛
    public int calPoints(String[] ops) {
        Stack<Integer> stack = new Stack<>();
        for (String op : ops) {
            if (op.equals("C")) {
                stack.pop();
            } else if (op.equals("D")) {
                stack.push(stack.peek() * 2);
            } else if (op.equals("+")) {
                int top = stack.pop();
                int newTop = top + stack.peek();
                stack.push(top);
                stack.push(newTop);
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
        int sCount = 0;
        int tIndex = t.length() - 1;
        int tCount = 0;
        while (sIndex >= 0 || tIndex >= 0) {
            while (sIndex >= 0) {
                if (s.charAt(sIndex) == '#') {
                    --sIndex;
                    ++sCount;
                } else if (sCount > 0) {
                    --sIndex;
                    --sCount;
                } else {
                    break;

                }

            }
            while (tIndex >= 0) {
                if (t.charAt(tIndex) == '#') {
                    --tIndex;
                    ++tCount;
                } else if (tCount > 0) {
                    --tIndex;
                    --tCount;
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
    // 剑指 Offer 31. 栈的压入、弹出序列
    // pushed = [1,2,3,4,5], popped = [4,5,3,2,1] true
    // pushed = [1,2,3,4,5], popped = [4,3,5,1,2] false
    public boolean validateStackSequences(int[] pushed, int[] popped) {
        int n = pushed.length;
        int j = 0;
        Stack<Integer> stack = new Stack<>();
        for (int push : pushed) {
            stack.push(push);
            while (!stack.isEmpty() && j < n && popped[j] == stack.peek()) {
                ++j;
                stack.pop();
            }
        }
        return stack.isEmpty();

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

    // 496. 下一个更大元素 (I Next Greater Element I)
    public int[] nextGreaterElement(int[] nums1, int[] nums2) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums1.length; ++i) {
            map.put(nums1[i], i);
        }
        int[] res = new int[nums1.length];
        Arrays.fill(res, -1);
        Stack<Integer> s = new Stack<>();
        for (int i = 0; i < nums2.length; ++i) {
            while (!s.isEmpty() && nums2[i] > nums2[s.peek()]) {
                int pop = nums2[s.pop()];
                if (map.containsKey(pop)) {
                    res[map.get(pop)] = nums2[i];
                }
            }
            s.push(i);
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

    // 503. 下一个更大元素 II (Next Greater Element II)
    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        int[] arr = new int[n * 2];
        for (int i = 0; i < n * 2; ++i) {
            arr[i] = nums[i];
            arr[i + n] = nums[i];
        }
        Arrays.fill(res, -1);
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < n * 2; ++i) {
            while (!stack.isEmpty() && stack.peek() < arr[i]) {
                res[stack.pop()] = arr[i % n];
            }
            stack.push(i);
        }
        return res;

    }

    // 1003. 检查替换后的词是否有效 (Check If Word Is Valid After Substitutions)
    // s = "abcabcababcc"
    public boolean isValid1003(String s) {
        int n = s.length();
        if (n % 3 != 0) {
            return false;
        }
        char[] arr = s.toCharArray();
        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < n; ++i) {
            char c = arr[i];
            if (c == 'c') {
                if (builder.length() <= 1 || builder.charAt(builder.length() - 1) != 'b'
                        || builder.charAt(builder.length() - 2) != 'a') {
                    return false;
                }
                builder.delete(builder.length() - 2, builder.length());
            } else {
                builder.append(c);
            }
        }
        return builder.isEmpty();

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

    // 1598. 文件夹操作日志搜集器 (Crawler Log Folder)
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

    // 1598. 文件夹操作日志搜集器 (Crawler Log Folder)
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

    // 150. 逆波兰表达式求值 // 剑指 Offer II 036. 后缀表达式
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

    // 150. 逆波兰表达式求值 // 剑指 Offer II 036. 后缀表达式
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

    // 735. 行星碰撞 (Asteroid Collision)
    // 剑指 Offer II 037. 小行星碰撞
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

    // 735. 行星碰撞 (Asteroid Collision)
    // 剑指 Offer II 037. 小行星碰撞
    public int[] asteroidCollision2(int[] asteroids) {
        Stack<Integer> stack = new Stack<>();
        search: for (int asteroid : asteroids) {
            while (!stack.isEmpty() && stack.peek() > 0 && asteroid < 0) {
                if (stack.peek() + asteroid < 0) {
                    stack.pop();
                } else if (stack.peek() + asteroid == 0) {
                    stack.pop();
                    continue search;
                } else if (stack.peek() + asteroid > 0) {
                    continue search;
                }
            }
            stack.push(asteroid);
        }
        int n = stack.size();
        int[] res = new int[n];
        while (!stack.isEmpty()) {
            res[--n] = stack.pop();
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

    // 316. 去除重复字母 (Remove Duplicate Letters)
    // 1081. 不同字符的最小子序列 (Smallest Subsequence of Distinct Characters)
    public String removeDuplicateLetters(String s) {
        int[] counts = new int[26];
        boolean[] seen = new boolean[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        StringBuilder res = new StringBuilder();
        for (char c : s.toCharArray()) {
            if (!seen[c - 'a']) {
                while (res.length() != 0 && c < res.charAt(res.length() - 1)) {
                    if (counts[res.charAt(res.length() - 1) - 'a'] > 0) {
                        seen[res.charAt(res.length() - 1) - 'a'] = false;
                        res.setLength(res.length() - 1);
                    } else {
                        break;
                    }
                }
                res.append(c);
                seen[c - 'a'] = true;
            }
            --counts[c - 'a'];
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

    // 136. 只出现一次的数字
    public int singleNumber(int[] nums) {
        int ans = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            ans ^= nums[i];
        }
        return ans;

    }

    // 137. 只出现一次的数字 II (Single Number II) // 剑指 Offer II 004. 只出现一次的数字
    public int singleNumberII(int[] nums) {
        int res = 0;
        for (int i = 0; i < 32; ++i) {
            int count = 0;
            for (int num : nums) {
                count += (num >> i) & 1;
            }
            res |= (count % 3) << i;

        }
        return res;

    }

    // 260. 只出现一次的数字 III (Single Number III)
    // 剑指 Offer 56 - I. 数组中数字出现的次数
    public int[] singleNumberIII(int[] nums) {
        int xor = 0;
        for (int x : nums) {
            xor ^= x;
        }
        xor &= -xor;
        int[] res = new int[2];
        for (int x : nums) {
            res[(xor & x) != 0 ? 1 : 0] ^= x;
        }
        return res;
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

    // 223. 矩形面积 (与836相似)
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

    // 836. 矩形重叠 (与223相似)
    public boolean isRectangleOverlap(int[] rec1, int[] rec2) {
        int maxLeft = Math.max(rec1[0], rec2[0]);
        int minRight = Math.min(rec1[2], rec2[2]);
        int minTop = Math.min(rec1[3], rec2[3]);
        int maxBottom = Math.max(rec1[1], rec2[1]);
        return (maxLeft < minRight) && (minTop > maxBottom);

    }

    // 299. 猜数字游戏（与“面试题 16.15. 珠玑妙算” 类似）
    public String getHint2(String secret, String guess) {
        int[] counts = new int[10];
        int bulls = 0;
        for (int i = 0; i < secret.length(); ++i) {
            if (secret.charAt(i) == guess.charAt(i)) {
                ++bulls;
            }
            ++counts[secret.charAt(i) - '0'];
        }
        int bullsAndCows = 0;
        for (char c : guess.toCharArray()) {
            if (counts[c - '0']-- > 0) {
                ++bullsAndCows;
            }
        }
        int cows = bullsAndCows - bulls;
        return bulls + "A" + cows + "B";

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

    // 386. 字典序排数 (Lexicographical Numbers)
    public List<Integer> lexicalOrder(int n) {
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i <= 9; ++i) {
            dfs386(i, n, res);
        }
        return res;
    }

    private void dfs386(int x, int limit, List<Integer> list) {
        if (x > limit) {
            return;
        }
        list.add(x);
        for (int i = 0; i <= 9; ++i) {
            dfs386(x * 10 + i, limit, list);
        }
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

    // 868. 二进制间距 (Binary Gap)
    public int binaryGap(int n) {
        int pre = 30;
        int res = 0;
        for (int x = n; x != 0; x &= x - 1) {
            int lb = Integer.numberOfTrailingZeros(x);
            res = Math.max(res, lb - pre);
            pre = lb;
        }
        return res;

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

    // 385. 迷你语法分析器 (Mini Parser) --stack
    public NestedInteger deserialize(String s) {
        if (s.charAt(0) != '[') {
            return new NestedInteger(Integer.parseInt(s));
        }
        Stack<NestedInteger> stack = new Stack<>();
        int index = 0;
        while (index < s.length()) {
            if (s.charAt(index) == '[') {
                stack.push(new NestedInteger());
            } else if (s.charAt(index) == '-' || Character.isDigit(s.charAt(index))) {
                int sign = 1;
                if (s.charAt(index) == '-') {
                    sign = -1;
                    ++index;
                }
                int num = 0;
                while (Character.isDigit(s.charAt(index))) {
                    num = num * 10 + (s.charAt(index++) - '0');
                }
                --index;
                stack.peek().add(new NestedInteger(num * sign));
            } else if (s.charAt(index) == ']') {
                if (stack.size() > 1) {
                    NestedInteger last = stack.pop();
                    stack.peek().add(last);
                }
            }
            ++index;
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

    // 636. 函数的独占时间 (Exclusive Time of Functions)
    public int[] exclusiveTime(int n, List<String> logs) {
        int[] res = new int[n];
        Stack<Integer> stack = new Stack<>();
        int pre = -1;
        for (String log : logs) {
            int id = Integer.parseInt(log.substring(0, log.indexOf(":")));
            int startOrEndTime = Integer.parseInt(log.substring(log.lastIndexOf(":") + 1));
            if (log.contains("start")) {
                if (!stack.isEmpty()) {
                    res[stack.peek()] += startOrEndTime - pre;
                }
                stack.push(id);
                pre = startOrEndTime;
            } else {
                int pop = stack.pop();
                res[pop] += startOrEndTime - pre + 1;
                pre = startOrEndTime + 1;
            }
        }
        return res;

    }

    // 1673. 找出最具竞争力的子序列 (Find the Most Competitive Subsequence)
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
    private int i1410;

    public String entityParser(String text) {
        StringBuilder res = new StringBuilder();
        while (i1410 < text.length()) {
            if (text.charAt(i1410) == '&') {
                char parse = parse(text.substring(i1410, text.length()));
                if (parse != ' ') {
                    res.append(parse);
                } else {
                    res.append(text.charAt(i1410));
                }
            } else {
                res.append(text.charAt(i1410));
            }
            ++i1410;
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
            i1410 += 5;
            return '\"';
        }
        if (string.indexOf("&apos;") == 0) {
            i1410 += 5;
            return '\'';
        }
        if (string.indexOf("&amp;") == 0) {
            i1410 += 4;
            return '&';
        }
        if (string.indexOf("&gt;") == 0) {
            i1410 += 3;
            return '>';
        }
        if (string.indexOf("&lt;") == 0) {
            i1410 += 3;
            return '<';
        }
        if (string.indexOf("&frasl;") == 0) {
            i1410 += 6;
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

    // 599. 两个列表的最小索引总和 (Minimum Index Sum of Two Lists)
    public String[] findRestaurant(String[] list1, String[] list2) {
        Map<String, Integer> map = new HashMap<>();
        for (int i = 0; i < list1.length; ++i) {
            map.put(list1[i], i);
        }
        List<String> res = new ArrayList<>();
        int minIndexSum = Integer.MAX_VALUE;
        for (int i = 0; i < list2.length; ++i) {
            if (map.containsKey(list2[i])) {
                if (map.get(list2[i]) + i < minIndexSum) {
                    minIndexSum = map.get(list2[i]) + i;
                    res.clear();
                    res.add(list2[i]);
                } else if (minIndexSum == map.get(list2[i]) + i) {
                    res.add(list2[i]);
                }
            }
        }
        return res.toArray(new String[res.size()]);

    }

    // 593. 有效的正方形 (Valid Square) --数学
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

    // 594. 最长和谐子序列 (Longest Harmonious Subsequence)
    public int findLHS(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int x : nums) {
            map.merge(x, 1, Integer::sum);
        }
        int res = 0;
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            if (map.get(entry.getKey() - 1) != null) {
                res = Math.max(res, entry.getValue() + map.getOrDefault(entry.getKey() - 1, 0));
            }
        }
        return res;

    }

    // 1905. 统计子岛屿 (Count Sub Islands) --bfs
    public int countSubIslands2(int[][] grid1, int[][] grid2) {
        int m = grid2.length;
        int n = grid2[0].length;
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid2[i][j] == 1) {
                    res += check1905(i, j, grid1, grid2) ? 1 : 0;
                }
            }
        }
        return res;

    }

    private boolean check1905(int i, int j, int[][] grid1, int[][] grid2) {
        int m = grid2.length;
        int n = grid2[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        boolean check = grid1[i][j] == 1;
        grid2[i][j] = 0;
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { i, j });
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid2[nx][ny] == 1) {
                    grid2[nx][ny] = 0;
                    queue.offer(new int[] { nx, ny });
                    if (grid1[nx][ny] == 0) {
                        check = false;
                    }
                }
            }
        }
        return check;
    }

    // 1905. 统计子岛屿 (Count Sub Islands) --并查集
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
                        if (grid1[i][j] == 0) {
                            not.add(root);
                            set.remove(root);
                        }
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

    // 1020. 飞地的数量 (Number of Enclaves) --bfs
    public int numEnclaves(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> q = new LinkedList<>();
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    int sum = 0;
                    boolean flag = false;
                    q.offer(new int[] { i, j });
                    while (!q.isEmpty()) {
                        int[] cur = q.poll();
                        int x = cur[0];
                        int y = cur[1];
                        if (x == 0 || x == m - 1 || y == 0 || y == n - 1) {
                            flag = true;
                        }
                        ++sum;
                        grid[x][y] = 0;
                        for (int[] d : directions) {
                            int nx = x + d[0];
                            int ny = y + d[1];
                            if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 1) {
                                grid[nx][ny] = 0;
                                q.offer(new int[] { nx, ny });
                            }
                        }
                    }
                    if (!flag) {
                        res += sum;
                    }
                }
            }
        }
        return res;
    }

    // 1020. 飞地的数量 (Number of Enclaves) --并查集
    public int numEnclaves2(int[][] grid) {
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
                    }
                    if (i + 1 < m && grid[i + 1][j] == 1) {
                        union.union(getIndex1020(n, i, j), getIndex1020(n, i + 1, j));
                    }
                    if (j + 1 < n && grid[i][j + 1] == 1) {
                        union.union(getIndex1020(n, i, j), getIndex1020(n, i, j + 1));
                    }
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1 && !union.isConnected(getIndex1020(n, i, j), dummy)) {
                    ++res;
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

    // 1020. 飞地的数量 (Number of Enclaves) --dfs
    private int m1020;
    private int n1020;
    private int[][] grid1020;
    private boolean flag1020;

    public int numEnclaves3(int[][] grid) {
        this.m1020 = grid.length;
        this.n1020 = grid[0].length;
        this.grid1020 = grid;
        int res = 0;
        for (int i = 0; i < m1020; ++i) {
            for (int j = 0; j < n1020; ++j) {
                if (grid[i][j] == 1) {
                    flag1020 = false;
                    int cur = dfs1020(i, j);
                    if (!flag1020) {
                        res += cur;
                    }
                }
            }
        }
        return res;

    }

    private int dfs1020(int i, int j) {
        if (i == 0 || i == m1020 - 1 || j == 0 || j == n1020 - 1) {
            flag1020 = true;
        }
        int cnt = 1;
        grid1020[i][j] = 0;
        int[][] dirs = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
        for (int[] d : dirs) {
            int nx = i + d[0];
            int ny = j + d[1];
            if (nx >= 0 && nx < m1020 && ny >= 0 && ny < n1020 && grid1020[nx][ny] == 1) {
                cnt += dfs1020(nx, ny);
            }
        }
        return cnt;
    }

    // 1254. 统计封闭岛屿的数目 (Number of Closed Islands) --bfs
    public int closedIsland(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> q = new LinkedList<>();
        int[][] dirs = { { 0, 1 }, { 0, -1 }, { -1, 0 }, { 1, 0 } };
        int res = 0;
        boolean flag = false;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0) {
                    flag = false;
                    q.offer(new int[] { i, j });
                    while (!q.isEmpty()) {
                        int[] cur = q.poll();
                        int x = cur[0];
                        int y = cur[1];
                        grid[x][y] = 1;
                        if (x == 0 || x == m - 1 || y == 0 || y == n - 1) {
                            flag = true;
                        }
                        for (int[] d : dirs) {
                            int nx = x + d[0];
                            int ny = y + d[1];
                            if (nx >= 0 && nx < m && ny >= 0 && ny < n && grid[nx][ny] == 0) {
                                q.offer(new int[] { nx, ny });
                            }
                        }
                    }
                    if (!flag) {
                        ++res;
                    }
                }
            }
        }
        return res;
    }

    // 1254. 统计封闭岛屿的数目 (Number of Closed Islands) --并查集
    public int closedIsland2(int[][] grid) {
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
                    }
                    if (i + 1 < m && grid[i + 1][j] == 0) {
                        union.union(getIndex1254(n, i, j), getIndex1254(n, i + 1, j));
                    }
                    if (j + 1 < n && grid[i][j + 1] == 0) {
                        union.union(getIndex1254(n, i, j), getIndex1254(n, i, j + 1));
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

    // 1254. 统计封闭岛屿的数目 (Number of Closed Islands) --dfs
    private boolean flag1254;
    private int m1254;
    private int n1254;
    private int[][] grid1254;
    private int[][] dirs1254 = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

    public int closedIsland3(int[][] grid) {
        int res = 0;
        this.m1254 = grid.length;
        this.n1254 = grid[0].length;
        this.grid1254 = grid;
        for (int i = 0; i < m1254; ++i) {
            for (int j = 0; j < n1254; ++j) {
                if (grid[i][j] == 0) {
                    flag1254 = false;
                    dfs1254(i, j);
                    if (!flag1254) {
                        ++res;
                    }
                }
            }
        }
        return res;
    }

    private void dfs1254(int i, int j) {
        if (i == 0 || i == m1254 - 1 || j == 0 || j == n1254 - 1) {
            flag1254 = true;
        }
        grid1254[i][j] = 1;
        for (int[] dir : dirs1254) {
            int nx = i + dir[0];
            int ny = j + dir[1];
            if (nx >= 0 && nx < m1254 && ny >= 0 && ny < n1254 && grid1254[nx][ny] == 0) {
                dfs1254(nx, ny);
            }
        }
    }

    // 1391. 检查网格中是否存在有效路径 (Check if There is a Valid Path in a Grid) --bfs
    public boolean hasValidPath(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        boolean[][] visited = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<>();
        queue.offer(new int[] { 0, 0 });
        visited[0][0] = true;
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            if (x == m - 1 && y == n - 1) {
                return true;
            }
            int val = grid[x][y];
            if (val == 1) {
                if (canAccessLeft(x, y, visited, grid)) {
                    visited[x][y - 1] = true;
                    queue.offer(new int[] { x, y - 1 });
                }
                if (canAccessRight(x, y, visited, grid)) {
                    visited[x][y + 1] = true;
                    queue.offer(new int[] { x, y + 1 });
                }
            } else if (val == 2) {
                if (canAccessUp(x, y, visited, grid)) {
                    visited[x - 1][y] = true;
                    queue.offer(new int[] { x - 1, y });
                }
                if (canAccessDown(x, y, visited, grid)) {
                    visited[x + 1][y] = true;
                    queue.offer(new int[] { x + 1, y });
                }
            } else if (val == 3) {
                if (canAccessLeft(x, y, visited, grid)) {
                    visited[x][y - 1] = true;
                    queue.offer(new int[] { x, y - 1 });
                }
                if (canAccessDown(x, y, visited, grid)) {
                    visited[x + 1][y] = true;
                    queue.offer(new int[] { x + 1, y });
                }
            } else if (val == 4) {
                if (canAccessRight(x, y, visited, grid)) {
                    visited[x][y + 1] = true;
                    queue.offer(new int[] { x, y + 1 });
                }
                if (canAccessDown(x, y, visited, grid)) {
                    visited[x + 1][y] = true;
                    queue.offer(new int[] { x + 1, y });
                }
            } else if (val == 5) {
                if (canAccessUp(x, y, visited, grid)) {
                    visited[x - 1][y] = true;
                    queue.offer(new int[] { x - 1, y });
                }
                if (canAccessLeft(x, y, visited, grid)) {
                    visited[x][y - 1] = true;
                    queue.offer(new int[] { x, y - 1 });
                }
            } else {
                if (canAccessUp(x, y, visited, grid)) {
                    visited[x - 1][y] = true;
                    queue.offer(new int[] { x - 1, y });
                }
                if (canAccessRight(x, y, visited, grid)) {
                    visited[x][y + 1] = true;
                    queue.offer(new int[] { x, y + 1 });
                }
            }
        }
        return false;

    }

    private boolean canAccessDown(int x, int y, boolean[][] visited, int[][] grid) {
        ++x;
        if (x >= grid.length) {
            return false;
        }
        if (visited[x][y]) {
            return false;
        }
        if (grid[x][y] == 2 || grid[x][y] == 5 || grid[x][y] == 6) {
            return true;
        }
        return false;
    }

    private boolean canAccessUp(int x, int y, boolean[][] visited, int[][] grid) {
        --x;
        if (x < 0) {
            return false;
        }
        if (visited[x][y]) {
            return false;
        }
        if (grid[x][y] == 2 || grid[x][y] == 3 || grid[x][y] == 4) {
            return true;
        }
        return false;
    }

    private boolean canAccessRight(int x, int y, boolean[][] visited, int[][] grid) {
        ++y;
        if (y >= grid[0].length) {
            return false;
        }
        if (visited[x][y]) {
            return false;
        }
        if (grid[x][y] == 1 || grid[x][y] == 3 || grid[x][y] == 5) {
            return true;
        }
        return false;
    }

    private boolean canAccessLeft(int x, int y, boolean[][] visited, int[][] grid) {
        --y;
        if (y < 0) {
            return false;
        }
        if (visited[x][y]) {
            return false;
        }
        if (grid[x][y] == 1 || grid[x][y] == 4 || grid[x][y] == 6) {
            return true;
        }
        return false;
    }

    // 1391. 检查网格中是否存在有效路径 (Check if There is a Valid Path in a Grid) --并查集

    // 1 表示连接左单元格和右单元格的街道。
    // 2 表示连接上单元格和下单元格的街道。
    // 3 表示连接左单元格和下单元格的街道。
    // 4 表示连接右单元格和下单元格的街道。
    // 5 表示连接左单元格和上单元格的街道。
    // 6 表示连接右单元格和上单元格的街道。

    public boolean hasValidPath2(int[][] grid) {
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
                    if (union.isConnected(index1, index2)) {
                        return true;
                    }
                    union.union(index1, index2);
                }
                if (j > 0 && grid[i][j - 1] == grid[i][j]) {
                    int index1 = getIndex1559(n, i, j);
                    int index2 = getIndex1559(n, i, j - 1);
                    if (union.isConnected(index1, index2)) {
                        return true;
                    }
                    union.union(index1, index2);
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

    // 面试题 16.19. 水域大小 (Pond Sizes LCCI) --bfs
    public int[] pondSizes(int[][] land) {
        int m = land.length;
        int n = land[0].length;
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 }, { 1, -1 }, { 1, 1 }, { -1, -1 }, { -1, 1 } };
        List<Integer> res = new ArrayList<>();
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (land[i][j] == 0) {
                    int count = 0;
                    queue.offer(new int[] { i, j });
                    land[i][j] = 1;
                    while (!queue.isEmpty()) {
                        int[] cur = queue.poll();
                        ++count;
                        int x = cur[0];
                        int y = cur[1];
                        for (int[] direction : directions) {
                            int nx = x + direction[0];
                            int ny = y + direction[1];
                            if (nx >= 0 && nx < m && ny >= 0 && ny < n && land[nx][ny] == 0) {
                                land[nx][ny] = 1;
                                queue.offer(new int[] { nx, ny });
                            }
                        }
                    }
                    res.add(count);
                }
            }
        }
        int[] ret = new int[res.size()];
        for (int i = 0; i < res.size(); ++i) {
            ret[i] = res.get(i);
        }
        Arrays.sort(ret);
        return ret;

    }

    // 面试题 16.19. 水域大小 (Pond Sizes LCCI) --并查集
    public int[] pondSizes2(int[][] land) {
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

    // 面试题 16.19. 水域大小
    private int m16_19;
    private int n16_19;
    private int[][] land16_19;

    public int[] pondSizes3(int[][] land) {
        this.m16_19 = land.length;
        this.n16_19 = land[0].length;
        this.land16_19 = land;
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < m16_19; ++i) {
            for (int j = 0; j < n16_19; ++j) {
                if (land[i][j] == 0) {
                    list.add(dfs16_19(i, j));
                }
            }
        }
        Collections.sort(list);
        int[] res = new int[list.size()];
        for (int i = 0; i < list.size(); ++i) {
            res[i] = list.get(i);
        }
        return res;

    }

    private int dfs16_19(int i, int j) {
        if (!(i >= 0 && i < m16_19 && j >= 0 && j < n16_19)) {
            return 0;
        }
        if (land16_19[i][j] != 0) {
            return 0;
        }
        int cnt = 1;
        land16_19[i][j] = 1;
        final int[][] dirs = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 }, { 1, 1 }, { 1, -1 }, { -1, 1 }, { -1, -1 } };
        for (int[] d : dirs) {
            int nx = i + d[0];
            int ny = j + d[1];
            cnt += dfs16_19(nx, ny);
        }
        return cnt;
    }

    // LCS 03. 主题空间 --bfs
    public int largestArea(String[] grid) {
        int m = grid.length;
        int n = grid[0].length();

        int[][] array = new int[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                array[i][j] = grid[i].charAt(j) - '0';
            }
        }
        boolean[][] visited = new boolean[m][n];
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i == 0 || i == m - 1 || j == 0 || j == n - 1 || array[i][j] == 0) {
                    queue.offer(new int[] { i, j });
                    visited[i][j] = true;
                }
            }
        }
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            int x = cur[0];
            int y = cur[1];
            for (int[] direction : directions) {
                int nx = x + direction[0];
                int ny = y + direction[1];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n && !visited[nx][ny]) {
                    if (array[x][y] == 0 || array[nx][ny] == array[x][y]) {
                        visited[nx][ny] = true;
                        queue.offer(new int[] { nx, ny });
                    }
                }
            }
        }
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!visited[i][j]) {
                    int curRes = 0;
                    queue.offer(new int[] { i, j });
                    visited[i][j] = true;
                    while (!queue.isEmpty()) {
                        ++curRes;
                        int[] cur = queue.poll();
                        int x = cur[0];
                        int y = cur[1];
                        for (int[] direction : directions) {
                            int nx = x + direction[0];
                            int ny = y + direction[1];
                            if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                                if (array[nx][ny] == array[x][y] && !visited[nx][ny]) {
                                    visited[nx][ny] = true;
                                    queue.offer(new int[] { nx, ny });
                                }
                            }
                        }
                    }
                    res = Math.max(res, curRes);
                }
            }
        }
        return res;
    }

    // LCS 03. 主题空间 --并查集
    public int largestArea2(String[] grid) {
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

    // 1653. 使字符串平衡的最少删除次数 (Minimum Deletions to Make String Balanced)
    public int minimumDeletions(String s) {
        int cntA = 0;
        for (char c : s.toCharArray()) {
            cntA += 'b' - c;
        }
        int res = cntA;
        int cur = cntA;
        for (char c : s.toCharArray()) {
            if (c == 'a') {
                --cur;
            } else {
                ++cur;
            }
            res = Math.min(res, cur);
        }
        return res;

    }

    // 1653. 使字符串平衡的最少删除次数 (Minimum Deletions to Make String Balanced) --动态规划
    public int minimumDeletions2(String s) {
        int res = 0;
        int cntB = 0;
        for (char c : s.toCharArray()) {
            if (c == 'b') {
                ++cntB;
            } else {
                res = Math.min(res + 1, cntB);
            }
        }
        return res;

    }

    // 451. 根据字符出现频率排序 (Sort Characters By Frequency)
    public String frequencySort(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray()) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }
        List<Map.Entry<Character, Integer>> list = new ArrayList<>(map.entrySet());
        Collections.sort(list, new Comparator<Map.Entry<Character, Integer>>() {

            @Override
            public int compare(Map.Entry<Character, Integer> o1, Map.Entry<Character, Integer> o2) {
                if (o1.getValue() == o2.getValue()) {
                    return o1.getKey() - o2.getKey();
                }
                return o2.getValue() - o1.getValue();
            }
        });
        StringBuilder builder = new StringBuilder();
        for (Map.Entry<Character, Integer> entry : list) {
            for (int i = 0; i < entry.getValue(); ++i) {
                builder.append(entry.getKey());
            }
        }
        return builder.toString();

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
            int count = counts[c];
            for (int i = 0; i < count; ++i) {
                res.append(c);
            }
        }
        return res.toString();

    }

    // 347. 前 K 个高频元素 (Top K Frequent Elements) (还需要掌握最小堆)
    // 剑指 Offer II 060. 出现频率最高的 k 个数字
    public int[] topKFrequent(int[] nums, int k) {
        // key--元素 value--频率
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
        for (int i = 0; i < res.length; ++i) {
            res[i] = list.get(i).getKey();
        }
        return res;
    }

    // 692. 前K个高频单词 (Top K Frequent Words)
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

    // 面试题 17.07. 婴儿名字 (Baby Names LCCI) --bfs
    public String[] trulyMostPopular2(String[] names, String[] synonyms) {
        Map<String, List<String>> graph = new HashMap<>();
        for (String synonym : synonyms) {
            int commaIndex = synonym.indexOf(",");
            String s1 = synonym.substring(1, commaIndex);
            String s2 = synonym.substring(commaIndex + 1, synonym.length() - 1);
            graph.computeIfAbsent(s1, k -> new ArrayList<>()).add(s2);
            graph.computeIfAbsent(s2, k -> new ArrayList<>()).add(s1);
        }

        List<String> res = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        Map<String, Integer> nameToCount = new HashMap<>();
        for (String name : names) {
            int bracketIndex = name.indexOf("(");
            String realName = name.substring(0, bracketIndex);
            int count = Integer.parseInt(name.substring(bracketIndex + 1, name.length() - 1));
            nameToCount.put(realName, count);
        }
        for (String name : nameToCount.keySet()) {
            if (!visited.contains(name)) {
                int curCount = nameToCount.get(name);
                String curName = name;
                visited.add(name);
                Queue<String> queue = new LinkedList<>();
                queue.add(name);
                while (!queue.isEmpty()) {
                    String cur = queue.poll();
                    if (graph.get(cur) == null) {
                        continue;
                    }
                    for (String neighbor : graph.get(cur)) {
                        if (!visited.contains(neighbor)) {
                            visited.add(neighbor);
                            queue.offer(neighbor);
                            if (nameToCount.containsKey(neighbor) && neighbor.compareTo(curName) < 0) {
                                curName = neighbor;
                            }
                            curCount += nameToCount.getOrDefault(neighbor, 0);
                        }
                    }
                }
                res.add(curName + "(" + curCount + ")");

            }
        }
        return res.toArray(new String[res.size()]);

    }

    // 面试题 17.07. 婴儿名字 (Baby Names LCCI) --并查集
    public String[] trulyMostPopular(String[] names, String[] synonyms) {
        int index = 0;
        Map<String, Integer> nameToIndex = new HashMap<>();
        Map<String, Integer> nameToCount = new HashMap<>();
        for (String name : names) {
            String[] split = name.split("\\(");
            nameToIndex.put(split[0], index++);
            nameToCount.put(split[0], Integer.parseInt(split[1].substring(0, split[1].length() - 1)));
        }
        for (String synonym : synonyms) {
            String[] split = synonym.split(",");
            String name1 = split[0].substring(1);
            if (!nameToIndex.containsKey(name1)) {
                nameToIndex.put(name1, index++);
                nameToCount.put(name1, 0);
            }
            String name2 = split[1].substring(0, split[1].length() - 1);
            if (!nameToIndex.containsKey(name2)) {
                nameToIndex.put(name2, index++);
                nameToCount.put(name2, 0);
            }
        }
        Union17_07 union = new Union17_07(index);
        for (String synonym : synonyms) {
            String[] split = synonym.split(",");
            String name1 = split[0].substring(1);
            String name2 = split[1].substring(0, split[1].length() - 1);
            union.union(nameToIndex.get(name1), nameToIndex.get(name2));
        }
        Map<Integer, List<String>> rootToNames = new HashMap<>();
        for (Map.Entry<String, Integer> entry : nameToIndex.entrySet()) {
            String name = entry.getKey();
            int i = entry.getValue();
            int root = union.getRoot(i);
            rootToNames.computeIfAbsent(root, k -> new ArrayList<>()).add(name);
        }
        String[] res = new String[rootToNames.size()];
        int resIndex = 0;
        for (List<String> list : rootToNames.values()) {
            int count = 0;
            int minIndex = 0;
            for (int j = 0; j < list.size(); ++j) {
                String name = list.get(j);
                count += nameToCount.get(name);
                if (name.compareTo(list.get(minIndex)) < 0) {
                    minIndex = j;
                }
            }
            res[resIndex++] = list.get(minIndex) + "(" + count + ")";
        }
        return res;

    }

    public class Union17_07 {
        private int[] parent;
        private int[] rank;

        public Union17_07(int n) {
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

    // 678. 有效的括号字符串
    public boolean checkValidString(String s) {
        int low = 0;
        int high = 0;
        for (char c : s.toCharArray()) {
            if (c == '(') {
                ++low;
                ++high;
            } else if (c == ')') {
                low = Math.max(0, low - 1);
                --high;
            } else {
                low = Math.max(0, low - 1);
                ++high;
            }
            if (high < 0) {
                return false;
            }
        }
        return low == 0;

    }

    // 1111. 有效括号的嵌套深度
    public int[] maxDepthAfterSplit2(String seq) {
        int a = 0;
        int b = 0;
        int[] res = new int[seq.length()];
        for (int i = 0; i < seq.length(); ++i) {
            if (seq.charAt(i) == '(') {
                if (a <= b) {
                    ++a;
                    res[i] = 0;
                } else {
                    ++b;
                    res[i] = 1;
                }
            } else {
                if (a >= b) {
                    --a;
                    res[i] = 0;
                } else {
                    --b;
                    res[i] = 1;
                }
            }
        }
        return res;

    }

    // 1111. 有效括号的嵌套深度
    public int[] maxDepthAfterSplit(String seq) {
        int depth = 0;
        int[] res = new int[seq.length()];
        for (int i = 0; i < seq.length(); ++i) {
            if (seq.charAt(i) == '(') {
                ++depth;
                res[i] = depth % 2;
            } else {
                res[i] = depth % 2;
                --depth;
            }
        }
        return res;

    }

    // 1006. 笨阶乘
    public int clumsy(int n) {
        int sign = 0;
        Stack<Integer> stack = new Stack<>();
        stack.push(n--);
        while (n > 0) {
            if (sign % 4 == 0) {
                stack.push(stack.pop() * n);
            } else if (sign % 4 == 1) {
                stack.push(stack.pop() / n);
            } else if (sign % 4 == 2) {
                stack.push(n);
            } else {
                stack.push(-n);
            }
            ++sign;
            --n;
        }
        int res = 0;
        while (!stack.isEmpty()) {
            res += stack.pop();
        }
        return res;

    }

    // 1472. 设计浏览器历史记录 (Design Browser History)
    class BrowserHistory {
        private List<String> history;
        private int ptr;

        public BrowserHistory(String homepage) {
            this.history = new ArrayList<>();
            history.add(homepage);
        }

        public void visit(String url) {
            history = history.subList(0, ptr + 1);
            history.add(url);
            ++ptr;
        }

        public String back(int steps) {
            ptr = Math.max(0, ptr - steps);
            return history.get(ptr);
        }

        public String forward(int steps) {
            ptr = Math.min(history.size() - 1, ptr + steps);
            return history.get(ptr);
        }
    }

    // 231. 2 的幂
    public boolean isPowerOfTwo(int n) {
        if (n <= 0) {
            return false;
        }
        int count = 0;
        while (n != 0) {
            count += n % 2;
            if (count > 1) {
                return false;
            }
            n /= 2;
        }
        return true;

    }

    // 231. 2 的幂
    public boolean isPowerOfTwo2(int n) {
        final int BIG = 1 << 30;
        return n > 0 && BIG % n == 0;
    }

    // 231. 2 的幂
    public boolean isPowerOfTwo3(int n) {
        return n > 0 && (n & (n - 1)) == 0;
    }

    // 231. 2 的幂
    public boolean isPowerOfTwo4(int n) {
        return n > 0 && (n & (-n)) == n;
    }

    // 342. 4的幂 (Power of Four)
    public boolean isPowerOfFour(int n) {
        return n > 0 && Integer.bitCount(n) == 1 && Integer.numberOfTrailingZeros(n) % 2 == 0;
    }

    // 326. 3的幂
    public boolean isPowerOfThree(int n) {
        if (n <= 0) {
            return false;
        }
        while (n % 3 == 0) {
            n /= 3;
        }
        return n == 1;

    }

    // 面试题 03.05. 栈排序
    class SortedStack {
        Stack<Integer> stack;
        Stack<Integer> assistStack;

        public SortedStack() {
            stack = new Stack<>();
            assistStack = new Stack<>();

        }

        public void push(int val) {
            while (!stack.isEmpty() && val > stack.peek()) {
                assistStack.push(stack.pop());
            }
            stack.push(val);
            while (!assistStack.isEmpty()) {
                stack.push(assistStack.pop());
            }

        }

        public void pop() {
            if (!stack.isEmpty()) {
                stack.pop();
            }
        }

        public int peek() {
            return stack.isEmpty() ? -1 : stack.peek();
        }

        public boolean isEmpty() {
            return stack.isEmpty();
        }
    }

    // 883. 三维形体投影面积
    public int projectionArea(int[][] grid) {
        int ans = 0;
        int n = grid.length;
        for (int i = 0; i < n; ++i) {
            int maxRow = 0;
            int maxCol = 0;
            for (int j = 0; j < n; ++j) {
                // 俯视图
                if (grid[i][j] > 0) {
                    ++ans;
                }
                // yOz面投影
                maxRow = Math.max(maxRow, grid[i][j]);
                // xOz面投影
                maxCol = Math.max(maxCol, grid[j][i]);

            }
            ans += maxRow + maxCol;
        }
        return ans;

    }

    // 205. 同构字符串 (类似290)
    public boolean isIsomorphic(String s, String t) {
        Map<Character, Character> map1 = new HashMap<>();
        Map<Character, Character> map2 = new HashMap<>();
        for (int i = 0; i < s.length(); ++i) {
            if ((map1.containsKey(s.charAt(i)) && map1.get(s.charAt(i)) != t.charAt(i))
                    || (map2.containsKey(t.charAt(i)) && map2.get(t.charAt(i)) != s.charAt(i))) {
                return false;
            }
            map1.put(s.charAt(i), t.charAt(i));
            map2.put(t.charAt(i), s.charAt(i));
        }
        return true;
    }

    // 290. 单词规律 (类似205)
    public boolean wordPattern(String pattern, String s) {
        String[] strings = s.split(" ");
        if (strings.length != pattern.length()) {
            return false;
        }
        Map<Character, String> map1 = new HashMap<>();
        Map<String, Character> map2 = new HashMap<>();
        for (int i = 0; i < pattern.length(); ++i) {
            if ((map1.containsKey(pattern.charAt(i)) && !map1.get(pattern.charAt(i)).equals(strings[i]))
                    || (map2.containsKey(strings[i]) && map2.get(strings[i]) != pattern.charAt(i))) {
                return false;
            }
            map1.put(pattern.charAt(i), strings[i]);
            map2.put(strings[i], pattern.charAt(i));
        }
        return true;

    }

    // 面试题 03.01. 三合一
    class TripleInOne {
        private int[] stack;
        private int perStackSize;
        private int peekIndex0;
        private int peekIndex1;
        private int peekIndex2;

        public TripleInOne(int stackSize) {
            stack = new int[stackSize * 3];
            perStackSize = stackSize;
            peekIndex0 = stackSize * 0 - 1;
            peekIndex1 = stackSize * 1 - 1;
            peekIndex2 = stackSize * 2 - 1;
        }

        public void push(int stackNum, int value) {
            if (perStackSize == 0) {
                return;
            }
            switch (stackNum) {
            // 第一个栈
            case 0:
                if (peekIndex0 != perStackSize - 1) {
                    stack[++peekIndex0] = value;
                }
                break;
            // 第二个栈
            case 1:
                if (peekIndex1 != perStackSize * 2 - 1) {
                    stack[++peekIndex1] = value;
                }
                break;
            // 第三个栈
            case 2:
                if (peekIndex2 != perStackSize * 3 - 1) {
                    stack[++peekIndex2] = value;
                }
                break;
            default:
                break;

            }
        }

        public int pop(int stackNum) {
            if (perStackSize == 0) {
                return -1;
            }
            switch (stackNum) {
            case 0:
                return peekIndex0 == perStackSize * 0 - 1 ? -1 : stack[peekIndex0--];
            case 1:
                return peekIndex1 == perStackSize * 1 - 1 ? -1 : stack[peekIndex1--];
            case 2:
                return peekIndex2 == perStackSize * 2 - 1 ? -1 : stack[peekIndex2--];
            default:
                return -1;
            }

        }

        public int peek(int stackNum) {
            if (perStackSize == 0) {
                return -1;
            }
            switch (stackNum) {
            case 0:
                return peekIndex0 == perStackSize * 0 - 1 ? -1 : stack[peekIndex0];
            case 1:
                return peekIndex1 == perStackSize * 1 - 1 ? -1 : stack[peekIndex1];
            case 2:
                return peekIndex2 == perStackSize * 2 - 1 ? -1 : stack[peekIndex2];
            default:
                return -1;

            }
        }

        public boolean isEmpty(int stackNum) {
            if (perStackSize == 0) {
                return true;
            }
            switch (stackNum) {
            case 0:
                return peekIndex0 == perStackSize * 0 - 1;
            case 1:
                return peekIndex1 == perStackSize * 1 - 1;
            case 2:
                return peekIndex2 == perStackSize * 2 - 1;
            default:
                return true;
            }
        }
    }

    // 455. 分发饼干
    public int findContentChildren(int[] g, int[] s) {
        Arrays.sort(g);
        Arrays.sort(s);
        int ans = 0;
        int i = 0;
        int j = 0;
        while (i < g.length && j < s.length) {
            if (s[j] >= g[i]) {
                ++ans;
                ++i;
            }
            ++j;
        }
        return ans;

    }

    // 812. 最大三角形面积 (鞋带公式)
    public double largestTriangleArea(int[][] points) {
        double ans = 0;
        int n = points.length;
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                for (int k = j + 1; k < n; ++k) {
                    ans = Math.max(ans, showlace(points[i], points[j], points[k]));
                }
            }
        }
        return ans;

    }

    private double showlace(int[] p1, int[] p2, int[] p3) {
        return 0.5 * Math
                .abs(p1[0] * p2[1] + p2[0] * p3[1] + p3[0] * p1[1] - p1[1] * p2[0] - p2[1] * p3[0] - p3[1] * p1[0]);
    }

    // 1030. 距离顺序排列矩阵单元格
    public int[][] allCellsDistOrder(int rows, int cols, int rCenter, int cCenter) {
        int[][] res = new int[rows * cols][2];
        int index = 0;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                res[index][0] = i;
                res[index][1] = j;
                ++index;
            }
        }

        Arrays.sort(res, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                int distance1 = Math.abs(o1[0] - rCenter) + Math.abs(o1[1] - cCenter);
                int distance2 = Math.abs(o2[0] - rCenter) + Math.abs(o2[1] - cCenter);
                return distance1 - distance2;
            }
        });
        return res;

    }

    // 1030. 距离顺序排列矩阵单元格 桶排序
    public int[][] allCellsDistOrder2(int rows, int cols, int rCenter, int cCenter) {
        List<List<int[]>> list = new ArrayList<>();
        int max = Math.max(rows, rows - rCenter - 1) + Math.max(cols, cols - cCenter - 1);
        for (int i = 0; i <= max; ++i) {
            list.add(new ArrayList<>());
        }
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                list.get(getManhattanDistance1030(i, j, rCenter, cCenter)).add(new int[] { i, j });
            }
        }
        int[][] res = new int[rows * cols][2];
        int index = 0;
        for (List<int[]> items : list) {
            for (int[] item : items) {
                res[index++] = item;
            }
        }
        return res;

    }

    private int getManhattanDistance1030(int x1, int y1, int x2, int y2) {
        return Math.abs(x1 - x2) + Math.abs(y1 - y2);
    }

    // 1828. 统计一个圆中点的数目
    public int[] countPoints(int[][] points, int[][] queries) {
        int[] res = new int[queries.length];
        int index = 0;
        for (int[] query : queries) {
            res[index++] = getCount1828(query[0], query[1], query[2], points);
        }
        return res;

    }

    private int getCount1828(int rCenter, int cCenter, int radius, int[][] points) {
        int count = 0;
        for (int[] point : points) {
            if ((point[0] - rCenter) * (point[0] - rCenter) + (point[1] - cCenter) * (point[1] - cCenter) <= radius
                    * radius) {
                ++count;
            }
        }
        return count;
    }

    // 388. 文件的最长绝对路径
    // "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"
    public int lengthLongestPath(String input) {
        Stack<Integer> stack = new Stack<>();
        int res = 0;
        int sum = 0;
        for (int i = 0; i < input.length(); ++i) {
            int k = 0;
            while (input.charAt(i) == '\t') {
                ++k;
                ++i;
            }
            while (stack.size() > k) {
                sum -= stack.pop();
            }
            int j = i;
            while (j < input.length() && input.charAt(j) != '\n') {
                ++j;
            }
            int len = j - i;
            stack.push(len);
            sum += len;
            if (input.substring(i, i + len).contains(".")) {
                res = Math.max(res, sum + stack.size() - 1);
            }
            i = j;
        }
        return res;

    }

    // 1526. 形成目标数组的子数组最少增加次数 (Minimum Number of Increments on Subarrays to Form a
    // Target Array)
    public int minNumberOperations(int[] target) {
        int ans = target[0];
        for (int i = 1; i < target.length; ++i) {
            ans += Math.max(target[i] - target[i - 1], 0);
        }
        return ans;
    }

    // 1106. 解析布尔表达式 (Parsing A Boolean Expression)
    public boolean parseBoolExpr(String expression) {
        Stack<Character> stack = new Stack<>();
        for (char c : expression.toCharArray()) {
            if (c == ',') {
                continue;
            }
            if (c != ')') {
                stack.push(c);
            } else {
                Stack<Character> subStack = new Stack<>();
                while (!stack.isEmpty() && stack.peek() != '(') {
                    subStack.push(stack.pop());
                }
                // 弹出“(”
                stack.pop();
                char res = getResult1106(subStack, stack.pop());
                stack.push(res);
            }
        }
        return stack.pop() == 't';
    }

    private char getResult1106(Stack<Character> subStack, char sign) {
        // “&”
        if (sign == '&') {
            while (!subStack.isEmpty()) {
                if (subStack.pop() == 'f') {
                    return 'f';
                }
            }
            return 't';
        }
        // “|”
        else if (sign == '|') {
            while (!subStack.isEmpty()) {
                if (subStack.pop() == 't') {
                    return 't';
                }
            }
            return 'f';
        }
        // “!”
        else {
            return subStack.pop() == 't' ? 'f' : 't';
        }
    }

    // 1106. 解析布尔表达式 (Parsing A Boolean Expression)
    public boolean parseBoolExpr2(String expression) {
        int n = expression.length();
        char c = expression.charAt(0);
        if ('t' == c) {
            return true;
        }
        if ('f' == c) {
            return false;
        }
        String sub = expression.substring(2, n - 1);
        if (c == '!') {
            return !parseBoolExpr(sub);
        }

        List<String> items = getItems1106(sub);
        if (c == '&') {
            for (String item : items) {
                if (!parseBoolExpr(item)) {
                    return false;
                }
            }
            return true;
        }

        for (String item : items) {
            if (parseBoolExpr(item)) {
                return true;
            }
        }
        return false;

    }

    private List<String> getItems1106(String s) {
        List<String> res = new ArrayList<>();
        int n = s.length();
        int i = 0;
        int j = 0;
        int balance = 0;
        while (j < n) {
            char c = s.charAt(j);
            if (c == ',' && balance == 0) {
                res.add(s.substring(i, j));
                i = j + 1;
            } else if (c == '(') {
                ++balance;
            } else if (c == ')') {
                --balance;
            }
            ++j;
        }
        res.add(s.substring(i, j));
        return res;
    }

    // 1704. 判断字符串的两半是否相似 (Determine if String Halves Are Alike)
    public boolean halvesAreAlike(String s) {
        int n = s.length();
        return getEvenLettersCount(s.substring(0, n / 2)) == getEvenLettersCount(s.substring(n / 2));

    }

    private int getEvenLettersCount(String s) {
        int count = 0;
        for (char c : s.toCharArray()) {
            if (judgeVowel1704(Character.toLowerCase(c))) {
                ++count;
            }
        }
        return count;
    }

    private boolean judgeVowel1704(char c) {
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
            return true;
        }
        return false;
    }

    // 869. 重新排序得到 2 的幂
    public boolean reorderedPowerOf2(int n) {
        int[] A = getCount869(n);
        for (int i = 0; i < 31; ++i) {
            if (Arrays.equals(A, getCount869(1 << i))) {
                return true;
            }
        }
        return false;

    }

    private int[] getCount869(int n) {
        int[] counts = new int[10];
        while (n > 0) {
            ++counts[n % 10];
            n /= 10;
        }
        return counts;
    }

    // 面试题 16.11. 跳水板
    public int[] divingBoard(int shorter, int longer, int k) {
        if (k == 0) {
            return new int[] {};
        }
        if (shorter == longer) {
            return new int[] { k * shorter };
        }
        int[] ans = new int[k + 1];
        for (int i = 0; i <= k; ++i) {
            ans[i] = longer * i + shorter * (k - i);
        }
        return ans;

    }

    // 面试题 16.02. 单词频率 --哈希表
    class WordsFrequency {
        Map<String, Integer> map;

        public WordsFrequency(String[] book) {
            map = new HashMap<>();
            for (String word : book) {
                map.put(word, map.getOrDefault(word, 0) + 1);
            }
        }

        public int get(String word) {
            return map.getOrDefault(word, 0);
        }
    }

    // 面试题 16.02. 单词频率 --字典树
    class WordsFrequency2 {
        private Trie16_02 trie;

        public WordsFrequency2(String[] book) {
            trie = new Trie16_02();
            for (String b : book) {
                trie.insert(b);
            }
        }

        public int get(String word) {
            return trie.getFrequency(word);
        }

        class Trie16_02 {
            private Trie16_02[] children;
            private int count;

            public Trie16_02() {
                children = new Trie16_02[26];
                count = 0;
            }

            public void insert(String word) {
                Trie16_02 node = this;
                for (char c : word.toCharArray()) {
                    int index = c - 'a';
                    if (node.children[index] == null) {
                        node.children[index] = new Trie16_02();
                    }
                    node = node.children[index];
                }
                ++node.count;
            }

            public int getFrequency(String word) {
                Trie16_02 node = this;
                for (char c : word.toCharArray()) {
                    int index = c - 'a';
                    if (node.children[index] == null) {
                        return 0;
                    }
                    node = node.children[index];
                }
                return node.count;
            }
        }
    }

    // 面试题 10.11. 峰与谷
    public void wiggleSort(int[] nums) {
        int[] arrCopy = nums.clone();
        Arrays.sort(arrCopy);
        int i = 0;
        int j = nums.length - 1;
        int index = 0;
        while (i < j) {
            nums[index++] = arrCopy[i];
            nums[index++] = arrCopy[j];
            ++i;
            --j;
        }
        if (nums.length % 2 == 1) {
            nums[index] = arrCopy[i];
        }

    }

    // 剑指 Offer 61. 扑克牌中的顺子
    public boolean isStraight(int[] nums) {
        int[] counts = new int[14];
        int max = -1;
        int min = 14;
        for (int num : nums) {
            if (num != 0) {
                max = Math.max(max, num);
                min = Math.min(min, num);
                if (++counts[num] > 1) {
                    return false;
                }
            }
        }
        return max - min < 5;

    }

    // 1717. 删除子字符串的最大得分 (Maximum Score From Removing Substrings)
    public int maximumGain(String s, int x, int y) {
        return Math.max(check1717(s + "_", x, y), check1717(new StringBuilder(s).reverse().toString() + "_", y, x));

    }

    private int check1717(String s, int x, int y) {
        int res = 0;
        int cntA = 0;
        int cntB = 0;
        for (char c : s.toCharArray()) {
            if (c == 'b') {
                if (cntA > 0) {
                    --cntA;
                    res += x;
                } else {
                    ++cntB;
                }
            } else if (c == 'a') {
                ++cntA;
            } else {
                res += Math.min(cntA, cntB) * y;
                cntA = cntB = 0;
            }
        }
        return res;
    }

    // 412. Fizz Buzz
    public List<String> fizzBuzz(int n) {
        List<String> res = new ArrayList<>();
        for (int i = 1; i <= n; ++i) {
            if (i % 5 == 0 && i % 3 == 0) {
                res.add("FizzBuzz");
            } else if (i % 5 == 0) {
                res.add("Buzz");
            } else if (i % 3 == 0) {
                res.add("Fizz");
            } else {
                res.add(String.valueOf(i));
            }
        }
        return res;

    }

    // 412. Fizz Buzz
    public List<String> fizzBuzz2(int n) {
        List<String> res = new ArrayList<>();
        for (int i = 1; i <= n; ++i) {
            StringBuilder builder = new StringBuilder();
            if (i % 3 == 0) {
                builder.append("Fizz");
            }
            if (i % 5 == 0) {
                builder.append("Buzz");
            }
            if (builder.length() == 0) {
                res.add(String.valueOf(i));
            } else {
                res.add(builder.toString());
            }
        }
        return res;

    }

    // 885. 螺旋矩阵 III
    public int[][] spiralMatrixIII(int rows, int cols, int rStart, int cStart) {
        int index = 1;
        int[][] res = new int[rows * cols][2];
        res[0] = new int[] { rStart, cStart };
        int[][] directions = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        int x = rStart;
        int y = cStart;
        int d = 0;
        int l = 1;
        while (index < rows * cols) {
            for (int i = 0; i < l; ++i) {
                x += directions[d][0];
                y += directions[d][1];
                if (x >= 0 && y >= 0 && x < rows && y < cols) {
                    res[index++] = new int[] { x, y };
                    if (index == rows * cols) {
                        return res;
                    }
                }
            }
            l += d % 2;
            d = (d + 1) % directions.length;
        }
        return res;

    }

    // 1518. 换水问题 (Water Bottles)
    public int numWaterBottles(int numBottles, int numExchange) {
        int res = numBottles;
        while (numBottles >= numExchange) {
            int exchange = numBottles / numExchange;
            res += exchange;
            numBottles = exchange + numBottles % numExchange;
        }
        return res;

    }

    // 1603. 设计停车系统 (Design Parking System)
    class ParkingSystem {
        private int big;
        private int medium;
        private int small;

        public ParkingSystem(int big, int medium, int small) {
            this.big = big;
            this.medium = medium;
            this.small = small;
        }

        public boolean addCar(int carType) {
            if (carType == 1) {
                return big-- > 0;
            } else if (carType == 2) {
                return medium-- > 0;
            } else {
                return small-- > 0;
            }
        }
    }

    // 1688. 比赛中的配对次数
    public int numberOfMatches(int n) {
        int res = 0;
        while (n > 0) {
            res += n / 2;
            n -= n / 2;
        }
        return res;
    }

    // 1688. 比赛中的配对次数
    public int numberOfMatches2(int n) {
        return n - 1;
    }

    // LCP 17. 速算机器人
    public int calculateLCP17(String s) {
        int x = 1;
        int y = 0;
        for (char c : s.toCharArray()) {
            if (c == 'A') {
                x = 2 * x + y;
            } else {
                y = 2 * y + x;
            }
        }
        return x + y;

    }

    // LCP 17. 速算机器人
    // 出现一个"A"，有 x+y=(2x+y)+y=2x+2y
    // 出现一个"B"，有 x+y=x+(2y+x)=2x+2y
    // 所以每出现一个A/B，都使x+y的值翻倍
    public int calculateLCP17_2(String s) {
        return 1 << s.length();
    }

    // 1860. 增长的内存泄露
    public int[] memLeak(int memory1, int memory2) {
        int time = 1;
        while (time <= Math.max(memory1, memory2)) {
            if (memory1 >= memory2) {
                memory1 -= time;
            } else {
                memory2 -= time;
            }
            ++time;
        }
        return new int[] { time, memory1, memory2 };
    }

    // 1103. 分糖果 II (Distribute Candies to People)
    public int[] distributeCandies(int candies, int num_people) {
        int[] res = new int[num_people];
        int i = 0;
        while (candies > 0) {
            int cnt = i + 1;
            res[i % num_people] += Math.min(cnt, candies);
            candies -= Math.min(cnt, candies);
            ++i;
        }
        return res;

    }

    // 1041. 困于环中的机器人
    // 满足以下条件之一，会构成死循环
    // 1、终点位于原点
    // 2、终点的方向与起点不同
    public boolean isRobotBounded(String instructions) {
        int[][] directions = { { 0, 1 }, { -1, 0 }, { 0, -1 }, { 1, 0 } };
        int d = 0;
        int x = 0;
        int y = 0;
        for (char instruction : instructions.toCharArray()) {
            if (instruction == 'G') {
                x += directions[d][0];
                y += directions[d][1];
            } else if (instruction == 'L') {
                d = (d - 1 + directions.length) % directions.length;
            } else {
                d = (d + 1) % directions.length;
            }
        }
        return (x == 0 && y == 0) || (d != 0);

    }

    // 498. 对角线遍历 (498. Diagonal Traverse)
    public int[] findDiagonalOrder(int[][] mat) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        int m = mat.length;
        int n = mat[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                map.computeIfAbsent(i + j, k -> new ArrayList<>()).add(mat[i][j]);
            }
        }

        List<Integer> res = new ArrayList<>();
        for (int k = 0; k < m + n - 1; ++k) {
            List<Integer> a = map.get(k);
            if (k % 2 == 0) {
                Collections.reverse(a);
            }
            res.addAll(a);
        }
        return res.stream().mapToInt(Integer::intValue).toArray();
    }

    // 1324. 竖直打印单词
    public List<String> printVertically(String s) {
        List<String> res = new ArrayList<>();
        String[] items = s.split(" ");
        int index = 0;
        while (true) {
            StringBuilder builder = new StringBuilder();
            for (String item : items) {
                builder.append(index < item.length() ? item.charAt(index) : " ");
            }
            while (builder.length() != 0 && Character.isWhitespace(builder.charAt(builder.length() - 1))) {
                builder.setLength(builder.length() - 1);
            }
            if (builder.length() == 0) {
                break;
            }
            res.add(builder.toString());
            ++index;
        }
        return res;
    }

    // 1324. 竖直打印单词
    public List<String> printVertically2(String s) {
        String[] strings = s.split("\\ ");
        List<String> list = new ArrayList<>();
        int index = 0;
        while (true) {
            StringBuilder builder = new StringBuilder();
            int min = Integer.MAX_VALUE;
            for (int i = 0; i < strings.length; ++i) {
                if (index < strings[i].length()) {
                    builder.append(strings[i].charAt(index));
                    min = i + 1;
                } else {
                    builder.append(" ");
                }
            }
            if (min == Integer.MAX_VALUE) {
                break;
            }
            list.add(builder.substring(0, min).toString());
            ++index;
        }
        return list;

    }

    // 1920. 基于排列构建数组
    public int[] buildArray(int[] nums) {
        int[] res = new int[nums.length];
        for (int i = 0; i < nums.length; ++i) {
            res[i] = nums[nums[i]];
        }
        return res;

    }

    // 68. 文本左右对齐
    public List<String> fullJustify(String[] words, int maxWidth) {
        List<String> list = new ArrayList<>();
        int i = 0;
        while (i < words.length) {
            int curWordLen = words[i].length();
            int j = i + 1;
            while (j < words.length && curWordLen + words[j].length() + j - i <= maxWidth) {
                curWordLen += words[j++].length();
            }
            int sumWhiteSpace = maxWidth - curWordLen;
            int slots = j - i - 1;
            StringBuilder builder = new StringBuilder();
            if (j == words.length || slots == 0) {
                while (i < j) {
                    builder.append(words[i++]).append(" ");
                }
                builder.setLength(builder.length() - 1);
                while (builder.length() < maxWidth) {
                    builder.append(" ");
                }
            } else {
                while (i < j) {
                    builder.append(words[i++]);
                    if (slots > 0) {
                        int count = 0;
                        if (sumWhiteSpace % slots == 0) {
                            count = sumWhiteSpace / slots;
                        } else {
                            count = (int) Math.ceil((double) sumWhiteSpace / slots);
                        }
                        sumWhiteSpace -= count;
                        --slots;
                        while (count-- > 0) {
                            builder.append(" ");
                        }
                    }
                }

            }
            list.add(builder.toString());
        }
        return list;
    }

    // 258. 各位相加
    public int addDigits(int num) {
        int res = num;
        while (res > 9) {
            int cur = 0;
            while (res > 0) {
                cur += res % 10;
                res /= 10;
            }
            res = cur;
        }
        return res;

    }

    // 258. 各位相加
    public int addDigits2(int num) {
        return (num - 1) % 9 + 1;
    }

    // 657. 机器人能否返回原点 (Robot Return to Origin)
    public boolean judgeCircle(String moves) {
        Map<Character, Integer> cnts = new HashMap<>();
        for (char c : moves.toCharArray()) {
            cnts.merge(c, 1, Integer::sum);
        }
        // 不可用 “==” 判断 因为 “==” 比较的是对象的引用地址，而不是值
        return cnts.getOrDefault('L', 0).equals(cnts.getOrDefault('R', 0))
                && cnts.getOrDefault('U', 0).equals(cnts.getOrDefault('D', 0));
    }

    // 592. 分数加减运算 (Fraction Addition and Subtraction)
    public String fractionAddition(String expression) {
        List<Character> sign = new ArrayList<>();
        List<Integer> num = new ArrayList<>();
        List<Integer> den = new ArrayList<>();
        for (int i = 0; i < expression.length(); ++i) {
            char c = expression.charAt(i);
            if (c == '-' || c == '+') {
                sign.add(c);
            } else {
                int left = i;
                int right = left + 1;
                while (right < expression.length() && expression.charAt(right) != '-'
                        && expression.charAt(right) != '+') {
                    ++right;
                }
                String[] split = expression.substring(left, right).split("/");
                num.add(Integer.parseInt(split[0]));
                den.add(Integer.parseInt(split[1]));
                i = right - 1;
            }
        }
        if (expression.charAt(0) != '-') {
            sign.add(0, '+');
        }
        int lcm = 1;
        for (int d : den) {
            lcm = getLCM592(lcm, d);
        }
        int res = 0;
        for (int i = 0; i < num.size(); ++i) {
            int curNum = lcm / den.get(i) * num.get(i);
            if (sign.get(i) == '+') {
                res += curNum;
            } else {
                res -= curNum;
            }
        }
        int x = getGCD592(Math.abs(res), Math.abs(lcm));
        return (res / x) + "/" + (lcm / x);
    }

    private int getLCM592(int a, int b) {
        return a * b / getGCD592(a, b);
    }

    private int getGCD592(int a, int b) {
        return b == 0 ? a : getGCD592(b, a % b);
    }

    // 1706. 球会落何处 (Where Will the Ball Fall)
    public int[] findBall(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int[] res = new int[n];
        for (int j = 0; j < n; ++j) {
            int curJ = j;
            for (int i = 0; i < m; ++i) {
                if (curJ == 0 && grid[i][curJ] == -1 || curJ == n - 1 && grid[i][curJ] == 1
                        || grid[i][curJ] == 1 && grid[i][curJ + 1] == -1
                        || grid[i][curJ] == -1 && grid[i][curJ - 1] == 1) {
                    curJ = -1;
                    break;
                }
                curJ += grid[i][curJ];
            }
            res[j] = curJ;
        }
        return res;

    }

    // 640. 求解方程 (Solve the Equation)
    public String solveEquation(String equation) {
        String[] strings = equation.split("=");
        int[] coefficient1 = getResult640(strings[0]);
        int[] coefficient2 = getResult640(strings[1]);
        if (coefficient1[0] == coefficient2[0]) {
            if (coefficient1[1] == coefficient2[1]) {
                return "Infinite solutions";
            } else {
                return "No solution";
            }
        }
        return "x=" + (coefficient2[1] - coefficient1[1]) / (coefficient1[0] - coefficient2[0]);

    }

    private int[] getResult640(String string) {
        List<Integer> sign = new ArrayList<>();
        for (char c : string.toCharArray()) {
            if (c == '+') {
                sign.add(1);
            } else if (c == '-') {
                sign.add(-1);
            }
        }
        if (string.charAt(0) != '-') {
            sign.add(0, 1);
        }
        int[] res = new int[2];
        int index = 0;
        for (String s1 : string.split("\\+")) {
            for (String s2 : s1.split("\\-")) {
                if (s2.isEmpty()) {
                    continue;
                }
                if (s2.charAt(s2.length() - 1) == 'x') {
                    if ("x".equals(s2)) {
                        res[0] += sign.get(index);
                    } else {
                        res[0] += Integer.parseInt(s2.substring(0, s2.length() - 1)) * sign.get(index);
                    }
                } else {
                    res[1] += Integer.parseInt(s2) * sign.get(index);
                }
                ++index;
            }
        }
        return res;
    }

    // 874. 模拟行走机器人 (Walking Robot Simulation)
    public int robotSim(int[] commands, int[][] obstacles) {
        final long M = (long) 1e5;
        Set<Long> set = new HashSet<>();
        for (int[] o : obstacles) {
            set.add(o[0] * M + o[1]);
        }
        int x = 0;
        int y = 0;
        int res = 0;
        int d = 0;
        int[][] dirs = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
        for (int c : commands) {
            if (c == -1) {
                d = (d + 1) % 4;
            } else if (c == -2) {
                d = (d - 1 + 4) % 4;
            } else {
                int step = c;
                while (step > 0) {
                    int nx = x + dirs[d][0];
                    int ny = y + dirs[d][1];
                    if (set.contains(nx * M + ny)) {
                        break;
                    }
                    x = nx;
                    y = ny;
                    res = Math.max(res, x * x + y * y);
                    --step;
                }
            }
        }
        return res;

    }

    // LCP 03. 机器人大冒险
    public boolean robot(String command, int[][] obstacles, int x, int y) {
        int curX = 0;
        int curY = 0;
        Set<Long> set = new HashSet<>();
        for (int[] obstacle : obstacles) {
            set.add(getRobotObstacles(obstacle[0], obstacle[1]));
        }
        int index = 0;
        while (curX <= x && curY <= y) {
            if (curX == x && curY == y) {
                return true;
            }
            char c = command.charAt(index++);
            index %= command.length();
            if (c == 'U') {
                ++curY;
            } else {
                ++curX;
            }
            if (set.contains(getRobotObstacles(curX, curY))) {
                return false;
            }

        }
        return false;

    }

    private long getRobotObstacles(long x, long y) {
        return x << 31 | y;
    }

    // LCP 02. 分式化简
    public int[] fraction(int[] cont) {
        int num = 0;
        int den = 1;
        for (int i = cont.length - 1; i >= 0; --i) {
            int temp = cont[i] * den + num;
            num = den;
            den = temp;
        }
        return new int[] { den, num };

    }

    // 1914. 循环轮转矩阵
    public int[][] rotateGrid(int[][] grid, int k) {
        int r1 = 0;
        int r2 = grid.length - 1;
        int c1 = 0;
        int c2 = grid[0].length - 1;
        while (r1 < r2 && c1 < c2) {
            int count = ((r2 - r1 + 1) + (c2 - c1 + 1)) * 2 - 4;
            int[] arr = new int[count];
            int index = 0;
            for (int c = c1; c <= c2; ++c) {
                arr[index++] = grid[r1][c];
            }
            for (int r = r1 + 1; r <= r2; ++r) {
                arr[index++] = grid[r][c2];
            }
            for (int c = c2 - 1; c >= c1; --c) {
                arr[index++] = grid[r2][c];
            }
            for (int r = r2 - 1; r >= r1 + 1; --r) {
                arr[index++] = grid[r][c1];
            }
            int[] res = new int[count];
            for (int i = 0; i < count; ++i) {
                res[(((i - k) % count) + count) % count] = arr[i];
            }
            index = 0;
            for (int c = c1; c <= c2; ++c) {
                grid[r1][c] = res[index++];
            }
            for (int r = r1 + 1; r <= r2; ++r) {
                grid[r][c2] = res[index++];
            }
            for (int c = c2 - 1; c >= c1; --c) {
                grid[r2][c] = res[index++];
            }
            for (int r = r2 - 1; r >= r1 + 1; --r) {
                grid[r][c1] = res[index++];
            }
            ++r1;
            --r2;
            ++c1;
            --c2;
        }
        return grid;
    }

    // 1806. 还原排列的最少操作步数 (Minimum Number of Operations to Reinitialize a
    // Permutation)
    public int reinitializePermutation(int n) {
        int i = 1;
        int res = 0;
        while (true) {
            ++res;
            i = (i & 1) == 1 ? n / 2 + (i - 1) / 2 : i / 2;
            if (i == 1) {
                return res;
            }
        }

    }

    // 1680. 连接连续二进制数字
    public int concatenatedBinary(int n) {
        final int MOD = 1000000007;
        int res = 0;
        int shift = 0;

        for (int i = 1; i <= n; ++i) {
            if ((i & (i - 1)) == 0) {
                ++shift;
            }
            res = (int) ((((long) res << shift) | i) % MOD);
        }
        return res;

    }

    // 1496. 判断路径是否相交
    public boolean isPathCrossing(String path) {
        Set<Integer> set = new HashSet<>();
        int x = 0;
        int y = 0;
        set.add(getCrossing1496(x, y));
        for (char c : path.toCharArray()) {
            if (c == 'N') {
                ++y;
            } else if (c == 'S') {
                --y;
            } else if (c == 'E') {
                ++x;
            } else {
                --x;
            }
            int num = getCrossing1496(x, y);
            if (set.contains(num)) {
                return true;
            }
            set.add(num);
        }
        return false;

    }

    private int getCrossing1496(int x, int y) {
        return x * 20001 + y;
    }

    // 1323. 6 和 9 组成的最大数字
    public int maximum69Number(int num) {
        char[] chars = String.valueOf(num).toCharArray();
        for (int i = 0; i < chars.length; ++i) {
            if (chars[i] == '6') {
                chars[i] = '9';
                break;
            }
        }
        return Integer.parseInt(String.valueOf(chars));

    }

    // 1433. 检查一个字符串是否可以打破另一个字符串
    public boolean checkIfCanBreak(String s1, String s2) {
        char[] chars1 = s1.toCharArray();
        char[] chars2 = s2.toCharArray();
        Arrays.sort(chars1);
        Arrays.sort(chars2);
        if (getChecked(chars1, chars2) || getChecked(chars2, chars1)) {
            return true;
        }
        return false;

    }

    private boolean getChecked(char[] chars1, char[] chars2) {
        for (int i = 0; i < chars1.length; ++i) {
            if (chars1[i] < chars2[i]) {
                return false;
            }
        }
        return true;
    }

    // 1599. 经营摩天轮的最大利润 (Maximum Profit of Operating a Centennial Wheel)
    public int minOperationsMaxProfit(int[] customers, int boardingCost, int runningCost) {
        int res = 0;
        int waiting = 0;
        int cost = 0;
        int maxCost = 0;
        int i = 0;
        int round = 0;
        int n = customers.length;
        while (i < n || waiting > 0) {
            ++round;
            if (i < n) {
                waiting += customers[i];
            }
            cost += Math.min(4, waiting) * boardingCost - runningCost;
            waiting -= Math.min(4, waiting);
            if (cost > maxCost) {
                maxCost = cost;
                res = round;
            }
            ++i;
        }
        return maxCost > 0 ? res : -1;
    }

    // 1583. 统计不开心的朋友
    public int unhappyFriends(int n, int[][] preferences, int[][] pairs) {
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for (int[] pair : pairs) {
            int index1 = pair[0];
            int index2 = pair[1];
            int[] preference1 = preferences[index1];
            for (int num : preference1) {
                if (num == index2) {
                    break;
                }
                map.computeIfAbsent(index1, k -> new HashSet<>()).add(num);
            }
            int[] preference2 = preferences[index2];
            for (int num : preference2) {
                if (num == index1) {
                    break;
                }
                map.computeIfAbsent(index2, k -> new HashSet<>()).add(num);
            }
        }
        int res = 0;
        for (int index : map.keySet()) {
            Set<Integer> set = map.get(index);
            if (set != null) {
                for (int subIndex : set) {
                    Set<Integer> subSet = map.get(subIndex);
                    if (subSet != null && subSet.contains(index)) {
                        ++res;
                        break;
                    }
                }
            }
        }
        return res;

    }

    // 1189. “气球” 的最大数量 (Maximum Number of Balloons)
    public int maxNumberOfBalloons(String text) {
        int[] counts = new int[26];
        for (char c : text.toCharArray()) {
            ++counts[c - 'a'];
        }
        int res = Integer.MAX_VALUE;
        if (counts['b' - 'a'] == 0) {
            return 0;
        }
        res = Math.min(res, counts['b' - 'a']);
        if (counts['a' - 'a'] == 0) {
            return 0;
        }
        res = Math.min(res, counts['a' - 'a']);
        if (counts['l' - 'a'] == 0) {
            return 0;
        }
        res = Math.min(res, counts['l' - 'a'] >> 1);
        if (counts['o' - 'a'] == 0) {
            return 0;
        }
        res = Math.min(res, counts['o' - 'a'] >> 1);
        if (counts['n' - 'a'] == 0) {
            return 0;
        }
        res = Math.min(res, counts['n' - 'a']);
        return res;

    }

    // 1221. 分割平衡字符串 (Split a String in Balanced Strings)
    public int balancedStringSplit(String s) {
        int res = 0;
        int L = 0;
        int R = 0;
        for (char c : s.toCharArray()) {
            if (c == 'L') {
                ++L;
            } else {
                ++R;
            }
            if (L == R) {
                ++res;
            }
        }
        return res;
    }

    // 1356. 根据数字二进制下 1 的数目排序 (Sort Integers by The Number of 1 Bits)
    public int[] sortByBits(int[] arr) {
        List<Integer> list = Arrays.stream(arr).boxed().collect(Collectors.toList());
        Collections.sort(list, (o1, o2) -> {
            int bitCount1 = Integer.bitCount(o1);
            int bitCount2 = Integer.bitCount(o2);
            if (bitCount1 == bitCount2) {
                return Integer.compare(o1, o2);
            }
            return Integer.compare(bitCount1, bitCount2);
        });
        return list.stream().mapToInt(i -> i).toArray();

    }

    // LCS 02. 完成一半题目
    public int halfQuestions(int[] questions) {
        int[] counts = new int[1001];
        for (int question : questions) {
            ++counts[question];
        }
        Arrays.sort(counts);
        int res = 0;
        int cur = 0;
        for (int i = counts.length - 1; i >= 0; --i) {
            if (cur >= questions.length / 2) {
                return res;
            }
            ++res;
            cur += counts[i];
        }
        return res;

    }

    // 1370. 上升下降字符串 (Increasing Decreasing String)
    public String sortString(String s) {
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        StringBuilder res = new StringBuilder();
        while (res.length() < s.length()) {
            for (int i = 0; i < counts.length; ++i) {
                if (counts[i]-- > 0) {
                    res.append((char) (i + 'a'));
                }
            }
            for (int i = counts.length - 1; i >= 0; --i) {
                if (counts[i]-- > 0) {
                    res.append((char) (i + 'a'));
                }
            }
        }
        return res.toString();

    }

    // 1790.仅执行一次字符串交换能否使两个字符串相等 (Check if One String Swap Can Make Strings Equal)
    public boolean areAlmostEqual(String s1, String s2) {
        int n = s1.length();
        int diff = 0;
        char a = '-';
        char b = '-';
        for (int i = 0; i < n; ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                if (++diff > 2) {
                    return false;
                }
                if (a == '-') {
                    a = s1.charAt(i);
                    b = s2.charAt(i);
                } else if (s1.charAt(i) != b || s2.charAt(i) != a) {
                    return false;
                }
            }
        }
        return diff != 1;

    }

    // 1400. 构造 K 个回文字符串 (Construct K Palindrome Strings)
    public boolean canConstruct(String s, int k) {
        int max = s.length();
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        int min = 0;
        for (int count : counts) {
            if (count % 2 == 1) {
                ++min;
            }
        }
        min = Math.max(1, min);
        return min <= k && k <= max;

    }

    // 1897. 重新分配字符使所有字符串都相等 (Redistribute Characters to Make All Strings Equal)
    public boolean makeEqual(String[] words) {
        int[] counts = new int[26];
        for (String word : words) {
            for (char c : word.toCharArray()) {
                ++counts[c - 'a'];
            }
        }
        for (int count : counts) {
            if (count % words.length != 0) {
                return false;
            }
        }
        return true;
    }

    // 767. 重构字符串 (Reorganize String)
    public String reorganizeString(String s) {
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        int maxIndex = 0;
        int maxCount = 0;
        for (int i = 0; i < counts.length; ++i) {
            if (counts[i] > maxCount) {
                maxIndex = i;
                maxCount = counts[i];
            }
            if (maxCount > (s.length() + 1) / 2) {
                return "";
            }
        }
        int index = 0;
        char[] res = new char[s.length()];
        while (counts[maxIndex]-- > 0) {
            res[index] = (char) (maxIndex + 'a');
            index += 2;
        }
        for (int i = 0; i < counts.length; ++i) {
            while (counts[i]-- > 0) {
                if (index >= res.length) {
                    index = 1;
                }
                res[index] = (char) (i + 'a');
                index += 2;
            }
        }
        return new String(res);

    }

    // 1781. 所有子字符串美丽值之和 (Sum of Beauty of All Substrings)
    public int beautySum(String s) {
        int res = 0;
        int n = s.length();
        for (int i = 0; i < n; ++i) {
            int[] counts = new int[26];
            int maxFreq = 0;
            for (int j = i; j < n; ++j) {
                maxFreq = Math.max(maxFreq, ++counts[s.charAt(j) - 'a']);
                int minFreq = n;
                for (int k = 0; k < 26; ++k) {
                    if (counts[k] != 0) {
                        minFreq = Math.min(minFreq, counts[k]);
                    }
                }
                res += maxFreq - minFreq;
            }
        }
        return res;

    }

    // 1941. 检查是否所有字符出现次数相同 (Check if All Characters Have Equal Number of
    // Occurrences)
    public boolean areOccurrencesEqual(String s) {
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        int c = 0;
        for (int count : counts) {
            if (count != 0) {
                if (c == 0) {
                    c = count;
                } else if (c != count) {
                    return false;
                }
            }
        }
        return true;

    }

    // 剑指 Offer 58 - II. 左旋转字符串
    public String reverseLeftWords(String s, int n) {
        StringBuilder res = new StringBuilder();
        for (int i = n; i < s.length(); ++i) {
            res.append(s.charAt(i));
        }
        for (int i = 0; i < n; ++i) {
            res.append(s.charAt(i));
        }
        return res.toString();
    }

    // 剑指 Offer 58 - II. 左旋转字符串
    public String reverseLeftWords2(String s, int n) {
        char[] res = new char[s.length()];
        for (int i = 0; i < s.length(); ++i) {
            res[(i - n + s.length()) % s.length()] = s.charAt(i);
        }
        return new String(res);
    }

    // 剑指 Offer 58 - II. 左旋转字符串
    public String reverseLeftWords3(String s, int n) {
        return s.substring(n) + s.substring(0, n);
    }

    // 1877. 数组中最大数对和的最小值 (Minimize Maximum Pair Sum in Array)
    public int minPairSum(int[] nums) {
        Arrays.sort(nums);
        int l = 0;
        int r = nums.length - 1;
        int min = 0;
        while (l < r) {
            min = Math.max(min, nums[l++] + nums[r--]);
        }
        return min;

    }

    // 1768.交替合并字符串 (Merge Strings Alternately)
    public String mergeAlternately(String word1, String word2) {
        StringBuilder res = new StringBuilder();
        int i = 0;
        int j = 0;
        while (i < word1.length() || j < word2.length()) {
            if (i < word1.length()) {
                res.append(word1.charAt(i++));
            }
            if (j < word2.length()) {
                res.append(word2.charAt(j++));
            }
        }
        return res.toString();

    }

    // 821. 字符的最短距离 (Shortest Distance to a Character)
    public int[] shortestToChar(String s, char c) {
        int prePos = -1;
        int[] res = new int[s.length()];
        Arrays.fill(res, Integer.MAX_VALUE);
        for (int i = 0; i < s.length(); ++i) {
            if (s.charAt(i) == c) {
                prePos = i;
            }
            if (prePos != -1) {
                res[i] = i - prePos;
            }
        }
        prePos = -1;
        for (int i = s.length() - 1; i >= 0; --i) {
            if (s.charAt(i) == c) {
                prePos = i;
            }
            if (prePos != -1) {
                res[i] = Math.min(res[i], prePos - i);
            }
        }
        return res;
    }

    // 1332. 删除回文子序列 (Remove Palindromic Subsequences)
    public int removePalindromeSub(String s) {
        if (s == null || s.length() == 0) {
            return 0;
        }
        if (new StringBuilder(s).reverse().toString().equals(s)) {
            return 1;
        }
        return 2;

    }

    // 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
    public int[] exchange(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            while (left < right && nums[left] % 2 == 1) {
                ++left;
            }
            while (left < right && nums[right] % 2 == 0) {
                --right;
            }
            int temp = nums[left];
            nums[left] = nums[right];
            nums[right] = temp;
            ++left;
            --right;
        }
        return nums;

    }

    // 696.计数二进制子串 (Count Binary Substrings)
    // "00110011"
    public int countBinarySubstrings(String s) {
        int res = 0;
        int index = 1;
        while (index < s.length()) {
            while (index < s.length() && s.charAt(index) == s.charAt(index - 1)) {
                ++index;
            }
            if (index >= s.length()) {
                break;
            }
            int left = index - 2;
            int right = index + 1;
            ++res;
            while (left >= 0 && right < s.length() && s.charAt(left) == s.charAt(left + 1)
                    && s.charAt(right) == s.charAt(right - 1)) {
                ++res;
                --left;
                ++right;
            }
            ++index;
        }
        return res;
    }

    // 917. 仅仅反转字母 (Reverse Only Letters)
    // Input: s = "a-bC-dEf-ghIj"
    // Output: "j-Ih-gfE-dCba"
    public String reverseOnlyLetters(String s) {
        int left = 0;
        int right = s.length() - 1;
        char[] chars = s.toCharArray();
        while (left < right) {
            while (left < right && !Character.isLetter(chars[left])) {
                ++left;
            }
            while (left < right && !Character.isLetter(chars[right])) {
                --right;
            }
            char temp = chars[right];
            chars[right] = chars[left];
            chars[left] = temp;
            ++left;
            --right;
        }
        return new String(chars);

    }

    // 925. 长按键入 (Long Pressed Name)
    public boolean isLongPressedName(String name, String typed) {
        int i = 0;
        int j = 0;
        while (j < typed.length()) {
            if (i < name.length() && name.charAt(i) == typed.charAt(j)) {
                ++i;
                ++j;
            } else if (j - 1 >= 0 && typed.charAt(j) == typed.charAt(j - 1)) {
                ++j;
            } else {
                return false;
            }
        }
        return i == name.length();

    }

    // 881. 救生艇 (Boats to Save People)
    public int numRescueBoats(int[] people, int limit) {
        Arrays.sort(people);
        int left = 0;
        int right = people.length - 1;
        int res = 0;
        while (left <= right) {
            if (people[left] + people[right] <= limit) {
                ++left;
            }
            --right;
            ++res;
        }
        return res;
    }

    // 986. 区间列表的交集 (Interval List Intersections)
    public int[][] intervalIntersection(int[][] firstList, int[][] secondList) {
        List<int[]> list = new ArrayList<>();
        int i = 0;
        int j = 0;
        while (i < firstList.length && j < secondList.length) {
            int high = Math.max(firstList[i][0], secondList[j][0]);
            int low = Math.min(firstList[i][1], secondList[j][1]);
            if (high <= low) {
                list.add(new int[] { high, low });
            }
            if (firstList[i][1] < secondList[j][1]) {
                ++i;
            } else {
                ++j;
            }
        }
        return list.toArray(new int[list.size()][]);

    }

    // 1023. 驼峰式匹配 (Camelcase Matching)
    public List<Boolean> camelMatch(String[] queries, String pattern) {
        List<Boolean> res = new ArrayList<>();
        for (String query : queries) {
            res.add(check1023(query, pattern));
        }
        return res;

    }

    private boolean check1023(String query, String pattern) {
        int m = query.length();
        int n = pattern.length();
        if (m < n) {
            return false;
        }
        int i = 0;
        int j = 0;
        while (i < m && j < n) {
            if (query.charAt(i) == pattern.charAt(j)) {
                ++i;
                ++j;
                continue;
            }
            if (Character.isUpperCase(query.charAt(i))) {
                return false;
            }
            ++i;
        }
        while (i < m && Character.isLowerCase(query.charAt(i))) {
            ++i;
        }
        return i == m && j == n;
    }

    // 剑指 Offer II 018. 有效的回文
    public boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        while (left < right) {
            while (left < right && !Character.isLetterOrDigit(s.charAt(left))) {
                ++left;
            }
            while (left < right && !Character.isLetterOrDigit(s.charAt(right))) {
                --right;
            }
            if (Character.toUpperCase(s.charAt(left)) != Character.toUpperCase(s.charAt(right))) {
                return false;
            }
            ++left;
            --right;
        }
        return true;

    }

    // 剑指 Offer 58 - I. 翻转单词顺序
    public String reverseWordsOffer58(String s) {
        String[] strings = s.split("\\s+");
        StringBuilder res = new StringBuilder();
        for (int i = strings.length - 1; i >= 0; --i) {
            if (!strings[i].trim().isEmpty()) {
                res.append(strings[i]).append(" ");
            }
        }
        return res.length() == 0 ? "" : res.substring(0, res.length() - 1).toString();
    }

    // 剑指 Offer 57 - II. 和为s的连续正数序列
    public int[][] findContinuousSequence(int target) {
        int left = 1;
        int right = 2;
        int sum = (left + right) * (right - left + 1) / 2;
        List<int[]> list = new ArrayList<>();
        while (left < right) {
            if (sum == target) {
                int[] sumAns = new int[right - left + 1];
                for (int k = left; k <= right; ++k) {
                    sumAns[k - left] = k;
                }
                list.add(sumAns);
                sum -= left;
                ++left;
            } else if (sum < target) {
                ++right;
                sum += right;
            } else {
                sum -= left;
                ++left;
            }
        }
        return list.toArray(new int[list.size()][]);

    }

    // 845. 数组中的最长山脉 (Longest Mountain in Array)
    public int longestMountain(int[] arr) {
        int n = arr.length;
        int res = 0;
        int i = 0;
        int j = 1;
        while (j < n) {
            boolean f1 = false;
            boolean f2 = false;
            while (j < n && arr[j] > arr[j - 1]) {
                f1 = true;
                ++j;
            }
            if (f1) {
                while (j < n && arr[j] < arr[j - 1]) {
                    f2 = true;
                    ++j;
                }
            }
            if (f1 && f2) {
                res = Math.max(res, j - i);
                i = j - 1;
            } else {
                i = j;
                ++j;
            }
        }
        return res;

    }

    // 763. 划分字母区间 (Partition Labels)
    public List<Integer> partitionLabels(String s) {
        int[] last = new int[26];
        for (int i = 0; i < s.length(); ++i) {
            last[s.charAt(i) - 'a'] = i;
        }
        int left = 0;
        int right = 0;
        List<Integer> res = new ArrayList<>();
        for (int i = 0; i < s.length(); ++i) {
            right = Math.max(right, last[s.charAt(i) - 'a']);
            if (i == right) {
                res.add(right - left + 1);
                left = right + 1;
            }
        }
        return res;

    }

    // LCP 06. 拿硬币
    public int minCount(int[] coins) {
        int res = 0;
        for (int coin : coins) {
            res += (coin + 1) / 2;
        }
        return res;

    }

    // 838. 推多米诺 (Push Dominoes)
    public String pushDominoes(String dominoes) {
        int n = dominoes.length();
        char[] arr = dominoes.toCharArray();
        int i = 0;
        int j = 0;
        while (j < n) {
            if (arr[j] == 'L') {
                if (arr[i] == 'R') {
                    Arrays.fill(arr, i + 1, i + (j - i - 1) / 2 + 1, 'R');
                    Arrays.fill(arr, j - (j - i - 1) / 2, j, 'L');
                } else {
                    Arrays.fill(arr, i, j, 'L');
                }
                i = j;
            } else if (arr[j] == 'R') {
                if (arr[i] == 'R') {
                    Arrays.fill(arr, i, j, 'R');
                }
                i = j;
            }
            ++j;
        }
        j = n - 1;
        while (j >= 0 && arr[j] == '.') {
            --j;
        }
        if (j >= 0 && arr[j] == 'R') {
            Arrays.fill(arr, j, n, 'R');
        }
        return String.valueOf(arr);

    }

    // 481. 神奇字符串 (Magical String)
    public int magicalString(int n) {
        int[] arr = new int[n + 2];
        arr[0] = 1;
        arr[1] = 2;
        arr[2] = 2;
        int i = 2;
        int j = 3;
        int c = 2;
        while (j < n) {
            c ^= 3;
            arr[j++] = c;
            if (arr[i] == 2) {
                arr[j++] = c;
            }
            ++i;
        }
        // 求1的前缀和，适用于多个查询
        // int[] prefix = new int[n + 3];
        // for (i = 1; i < prefix.length; ++i) {
        // prefix[i] = prefix[i - 1] + 2 - arr[i - 1];
        // }
        // return prefix[n];
        int res = 0;
        for (i = 0; i < n; ++i) {
            res += 2 - arr[i];
        }
        return res;

    }

    // 1945. 字符串转化后的各位数字之和 (Sum of Digits of String After Convert)
    public int getLucky(String s, int k) {
        int num = 0;
        for (char c : s.toCharArray()) {
            int cur = c - 'a' + 1;
            while (cur > 0) {
                num += cur % 10;
                cur /= 10;
            }
        }
        while (--k > 0 && num >= 10) {
            int cur = 0;
            while (num > 0) {
                cur += num % 10;
                num /= 10;
            }
            num = cur;
        }
        return num;

    }

    // 1903. 字符串中的最大奇数 (Largest Odd Number in String)
    public String largestOddNumber(String num) {
        int index = num.length() - 1;
        while (index >= 0) {
            if ((num.charAt(index) - '0') % 2 == 1) {
                return num.substring(0, index + 1);
            }
            --index;
        }
        return "";

    }

    // 1679. K 和数对的最大数目 (Max Number of K-Sum Pairs)
    public int maxOperations(int[] nums, int k) {
        Arrays.sort(nums);
        int left = 0;
        int right = nums.length - 1;
        int res = 0;
        while (left < right) {
            int sum = nums[left] + nums[right];
            if (sum == k) {
                ++left;
                --right;
                ++res;
            } else if (sum < k) {
                ++left;
            } else {
                --right;
            }
        }
        return res;

    }

    // 1850. 邻位交换的最小次数 (Minimum Adjacent Swaps to Reach the Kth Smallest Number)
    public int getMinSwaps(String num, int k) {
        char[] chars = num.toCharArray();
        char[] charsK = chars.clone();
        for (int i = 0; i < k; ++i) {
            getNextPermutation(charsK);
        }
        int res = 0;
        for (int i = 0; i < charsK.length; ++i) {
            if (charsK[i] != chars[i]) {
                for (int j = i + 1; j < charsK.length; ++j) {
                    if (charsK[j] == chars[i]) {
                        for (int m = j - 1; m >= i; --m) {
                            ++res;
                            swap1850(charsK, m, m + 1);
                        }
                        break;
                    }
                }
            }
        }
        return res;

    }

    private void getNextPermutation(char[] charsK) {
        int i = charsK.length - 1;
        while (i > 0) {
            if (charsK[i - 1] < charsK[i]) {
                break;
            }
            --i;
        }
        if (i == 0) {
            return;
        }
        --i;
        int j = charsK.length - 1;
        while (i < j) {
            if (charsK[i] < charsK[j]) {
                swap1850(charsK, i, j);
                break;
            }
            --j;
        }
        Arrays.sort(charsK, i + 1, charsK.length);
    }

    private void swap1850(char[] chars, int i, int j) {
        char temp = chars[i];
        chars[i] = chars[j];
        chars[j] = temp;
    }

    // 剑指 Offer II 072. 求平方根
    public int mySqrt(int x) {
        int left = 0;
        int right = x;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if ((long) mid * mid == x) {
                return mid;
            } else if ((long) mid * mid < x) {
                left = mid + 1;
            } else if ((long) mid * mid > x) {
                right = mid - 1;
            }
        }
        return right;

    }

    // 861. 翻转矩阵后的得分 (Score After Flipping Matrix)
    public int matrixScore(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        int res = m * (1 << (n - 1));
        for (int j = 1; j < n; ++j) {
            int col = 0;
            for (int i = 0; i < m; ++i) {
                if (grid[i][0] == 1) {
                    col += grid[i][j];
                } else {
                    col += 1 - grid[i][j];
                }
            }
            col = Math.max(col, m - col);
            res += col * (1 << (n - j - 1));
        }
        return res;

    }

    // 524. 通过删除字母匹配到字典里最长单词 (Longest Word in Dictionary through Deleting)
    public String findLongestWord(String s, List<String> dictionary) {
        String res = "";
        for (String dic : dictionary) {
            if (isSub524(s, dic)) {
                if (dic.length() > res.length() || (dic.length() == res.length() && dic.compareTo(res) < 0)) {
                    res = dic;
                }
            }
        }
        return res;

    }

    private boolean isSub524(String s, String dic) {
        int i = 0;
        int j = 0;
        while (i < s.length() && j < dic.length()) {
            if (s.charAt(i) == dic.charAt(j)) {
                ++j;
            }
            ++i;
        }
        return j == dic.length();
    }

    // 1048. 最长字符串链 (Longest String Chain)
    public int longestStrChain(String[] words) {
        Arrays.sort(words, (o1, o2) -> o1.length() - o2.length());
        int[] dp = new int[words.length];
        for (int i = 0; i < words.length; ++i) {
            for (int j = i + 1; j < words.length; ++j) {
                if (isPredecessor1048(words[i], words[j])) {
                    dp[j] = Math.max(dp[j], dp[i] + 1);
                }
            }
        }
        return Arrays.stream(dp).max().getAsInt() + 1;

    }

    private boolean isPredecessor1048(String string1, String string2) {
        if (string2.length() - string1.length() != 1) {
            return false;
        }
        int i = 0;
        int j = 0;
        int flag = 0;
        while (i < string1.length() && j < string2.length()) {
            if (string1.charAt(i) == string2.charAt(j)) {
                ++i;
            } else if (++flag > 1) {
                return false;
            }
            ++j;
        }
        return true;
    }

    // 1048. 最长字符串链 (Longest String Chain)
    private Map<String, Integer> memo1048;

    public int longestStrChain2(String[] words) {
        memo1048 = new HashMap<>();
        for (String s : words) {
            memo1048.put(s, 0);
        }
        int res = 0;
        for (String word : words) {
            res = Math.max(res, dfs1048(word));
        }
        return res;
    }

    private int dfs1048(String s) {
        if (memo1048.get(s) != 0) {
            return memo1048.get(s);
        }
        int res = 1;
        for (int i = 0; i < s.length(); ++i) {
            String next = s.substring(0, i) + s.substring(i + 1);
            if (memo1048.containsKey(next)) {
                res = Math.max(res, dfs1048(next) + 1);
            }
        }
        memo1048.put(s, res);
        return res;
    }

    // 1750. 删除字符串两端相同字符后的最短长度 (Minimum Length of String After Deleting Similar
    // Ends)
    public int minimumLength(String s) {
        int n = s.length();
        int i = 0;
        int j = n - 1;
        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) {
                break;
            }
            char c = s.charAt(i);
            while (i < j && s.charAt(i) == c) {
                ++i;
            }
            if (i == j) {
                return 0;
            }
            while (i < j && s.charAt(j) == c) {
                --j;
            }
        }
        return j - i + 1;

    }

    // 1963. 使字符串平衡的最小交换次数 (Minimum Number of Swaps to Make the String Balanced)
    public int minSwaps(String s) {
        int res = 0;
        int leftCount = 0;
        for (char c : s.toCharArray()) {
            if (c == '[') {
                ++leftCount;
            } else {
                --leftCount;
            }
            if (leftCount < 0) {
                ++res;
                leftCount += 2;
            }
        }
        return res;

    }

    // 826. 安排工作以达到最大收益 (Most Profit Assigning Work)
    public int maxProfitAssignment(int[] difficulty, int[] profit, int[] worker) {
        int n = difficulty.length;
        int[][] arr = new int[n][2];
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for (int i = 0; i < n; ++i) {
            arr[i][0] = difficulty[i];
            arr[i][1] = profit[i];
            map.merge(profit[i], 1, Integer::sum);
        }
        Arrays.sort(arr, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                return Integer.compare(o1[0], o2[0]);
            }

        });

        Arrays.sort(worker);
        int res = 0;
        int j = n - 1;
        for (int i = worker.length - 1; i >= 0; --i) {
            while (j >= 0 && arr[j][0] > worker[i]) {
                map.merge(arr[j][1], -1, Integer::sum);
                if (map.get(arr[j][1]) == 0) {
                    map.remove(arr[j][1]);
                }
                --j;
            }
            if (map.size() > 0) {
                res += map.lastKey();
            }
        }
        return res;

    }

    // 剑指 Offer 17. 打印从1到最大的n位数
    public int[] printNumbers(int n) {
        int count = (int) Math.pow(10, n) - 1;
        int[] res = new int[count];
        for (int i = 0; i < res.length; ++i) {
            res[i] = i + 1;
        }
        return res;

    }

    // 1813. 句子相似性 III (Sentence Similarity III)
    public boolean areSentencesSimilar(String sentence1, String sentence2) {
        String[] chars1 = sentence1.split("\\s+");
        String[] chars2 = sentence2.split("\\s+");
        if (chars1.length > chars2.length) {
            String[] temp = chars1;
            chars1 = chars2;
            chars2 = temp;
        }
        return isSimilar1813(chars1, chars2);

    }

    private boolean isSimilar1813(String[] chars1, String[] chars2) {
        int left = 0;
        while (left < chars1.length && chars1[left].equals(chars2[left])) {
            ++left;
        }
        if (left == chars1.length) {
            return true;
        }
        int right1 = chars1.length - 1;
        int right2 = chars2.length - 1;
        while (right1 >= 0 && chars1[right1].equals(chars2[right2])) {
            --right1;
            --right2;
        }
        if (right1 < 0) {
            return true;
        }

        return (left - right1) == 1;
    }

    // 1793. 好子数组的最大分数 (Maximum Score of a Good Subarray) --单调栈
    public int maximumScore(int[] nums, int k) {
        int n = nums.length;
        Stack<Integer> st = new Stack<>();
        int[] left = new int[n];
        Arrays.fill(left, -1);
        for (int i = 0; i < n; ++i) {
            while (!st.isEmpty() && nums[st.peek()] >= nums[i]) {
                st.pop();
            }
            if (!st.isEmpty()) {
                left[i] = st.peek();
            }
            st.push(i);
        }
        st.clear();
        int[] right = new int[n];
        Arrays.fill(right, n);
        for (int i = n - 1; i >= 0; --i) {
            while (!st.isEmpty() && nums[st.peek()] >= nums[i]) {
                st.pop();
            }
            if (!st.isEmpty()) {
                right[i] = st.peek();
            }
            st.push(i);
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            if (left[i] < k && k < right[i]) {
                res = Math.max(res, nums[i] * (right[i] - left[i] - 1));
            }
        }
        return res;
    }

    // LCP 18. 早餐组合
    public int breakfastNumber(int[] staple, int[] drinks, int x) {
        final int MOD = 1000000007;
        Arrays.sort(staple);
        Arrays.sort(drinks);
        int i = 0;
        int j = drinks.length - 1;
        int res = 0;
        while (i < staple.length && j >= 0) {
            if (staple[i] + drinks[j] > x) {
                --j;
            } else {
                res = (res + j + 1) % MOD;
                ++i;
            }
        }
        return res;

    }

    // LCP 28. 采购方案
    public int purchasePlans(int[] nums, int target) {
        final int MOD = 1000000007;
        Arrays.sort(nums);
        int res = 0;
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int sum = nums[left] + nums[right];
            if (sum > target) {
                --right;
            } else {
                res = (res + right - left) % MOD;
                ++left;
            }
        }
        return res;

    }

    // 1753. 移除石子的最大得分 (Maximum Score From Removing Stones)
    public int maximumScore(int a, int b, int c) {
        int[] arr = new int[] { a, b, c };
        Arrays.sort(arr);
        if (arr[0] + arr[1] <= arr[2]) {
            return arr[0] + arr[1];
        } else {
            return Arrays.stream(arr).sum() / 2;
        }

    }

    // 477. 汉明距离总和 (Total Hamming Distance)
    public int totalHammingDistance(int[] nums) {
        int res = 0;
        for (int i = 0; i < 30; ++i) {
            int c = 0;
            for (int num : nums) {
                c += (num >> i) & 1;
            }
            res += c * (nums.length - c);
        }
        return res;

    }

    // 293. 翻转游戏 (Flip Game) (plus)
    public List<String> generatePossibleNextMoves(String currentState) {
        List<String> res = new ArrayList<>();
        for (int i = 0; i < currentState.length() - 1; ++i) {
            if (currentState.charAt(i) == '+' && currentState.charAt(i + 1) == '+') {
                StringBuilder builder = new StringBuilder(currentState);
                builder.replace(i, i + 2, "--");
                res.add(builder.toString());
            }
        }
        return res;

    }

    // 1863.找出所有子集的异或总和再求和 (Sum of All Subset XOR Totals)
    public int subsetXORSum(int[] nums) {
        int n = nums.length;
        int res = 0;
        for (int num : nums) {
            res |= num;
        }
        return res << (n - 1);

    }

    // 1720. 解码异或后的数组 (Decode XORed Array)
    public int[] decode(int[] encoded, int first) {
        int[] res = new int[encoded.length + 1];
        res[0] = first;
        for (int i = 1; i < res.length; ++i) {
            res[i] = res[i - 1] ^ encoded[i - 1];
        }
        return res;
    }

    // 693. 交替位二进制数 (Binary Number with Alternating Bits)
    public boolean hasAlternatingBits(int n) {
        int xor = (n ^ (n >> 1)) + 1;
        return (xor & (xor - 1)) == 0;
    }

    // 693. 交替位二进制数 (Binary Number with Alternating Bits)
    public boolean hasAlternatingBits2(int n) {
        return (n ^ (n >> 1)) == (1 << (32 - Integer.numberOfLeadingZeros(n))) - 1;
    }

    // 762. 二进制表示中质数个计算置位 (Prime Number of Set Bits in Binary Representation)
    public int countPrimeSetBits(int left, int right) {
        int res = 0;
        for (int i = left; i <= right; ++i) {
            int count = Integer.bitCount(i);
            if (isPrime(count)) {
                ++res;
            }
        }
        return res;

    }

    private boolean isPrime(int count) {
        switch (count) {
        case 2:
        case 3:
        case 5:
        case 7:
        case 11:
        case 13:
        case 17:
        case 19:
            return true;
        default:
            return false;
        }
    }

    // 401. 二进制手表 (Binary Watch)
    public List<String> readBinaryWatch(int turnedOn) {
        List<String> res = new ArrayList<>();
        for (int h = 0; h < 12; ++h) {
            for (int m = 0; m < 60; ++m) {
                if (Integer.bitCount(h) + Integer.bitCount(m) == turnedOn) {
                    res.add(h + ":" + (m < 10 ? "0" : "") + m);
                }
            }
        }
        return res;

    }

    // 401. 二进制手表 (Binary Watch)
    public List<String> readBinaryWatch2(int turnedOn) {
        List<String> res = new ArrayList<>();
        for (int i = 0; i < 1024; ++i) {
            int h = i >> 6;
            int m = i & 0b111111;
            if (h < 12 && m < 60 && (Integer.bitCount(h) + Integer.bitCount(m)) == turnedOn) {
                res.add(h + ":" + (m < 10 ? "0" : "") + m);
            }
        }
        return res;

    }

    // 338. 比特位计数 (Counting Bits)
    // 剑指 Offer II 003. 前 n 个数字二进制中 1 的个数
    public int[] countBits(int n) {
        int[] res = new int[n + 1];
        for (int i = 1; i <= n; ++i) {
            res[i] = res[i & (i - 1)] + 1;
        }
        return res;
    }

    // 1342. 将数字变成 0 的操作次数 (Number of Steps to Reduce a Number to Zero)
    public int numberOfSteps(int num) {
        if (num == 0) {
            return 0;
        }
        int bt = Integer.bitCount(num);
        int zero = 31 - Integer.numberOfLeadingZeros(num);
        return bt + zero;
    }

    // 剑指 Offer 15. 二进制中1的个数
    public int hammingWeight(int n) {
        return Integer.bitCount(n);
    }

    // 剑指 Offer 15. 二进制中1的个数
    public int hammingWeight2(int n) {
        int res = 0;
        while (n != 0) {
            n &= n - 1;
            ++res;
        }
        return res;
    }

    // 剑指 Offer 15. 二进制中1的个数
    public int hammingWeight3(int n) {
        int res = 0;
        for (int i = 0; i < 32; ++i) {
            if (((n >> i) & 1) != 0) {
                ++res;
            }
        }
        return res;
    }

    // 190. 颠倒二进制位 (Reverse Bits)
    public int reverseBits(int n) {
        int res = 0;
        int i = 0;
        while (i++ < 32) {
            res <<= 1;
            res |= n & 1;
            n >>= 1;
        }
        return res;

    }

    // 190. 颠倒二进制位 (Reverse Bits)
    public int reverseBits2(int n) {
        return Integer.reverse(n);

    }

    // 面试题 05.07. 配对交换 (Exchange LCCI)
    public int exchangeBits(int num) {
        return ((num & 0x55555555) << 1) | ((num & 0xAAAAAAAA) >> 1);
    }

    // 1525. 字符串的好分割数目 (Number of Good Ways to Split a String)
    public int numSplits(String s) {
        Set<Character> set = new HashSet<>();
        int[] left = new int[s.length()];
        left[0] = 1;
        set.add(s.charAt(0));
        for (int i = 1; i < s.length(); ++i) {
            if (set.add(s.charAt(i))) {
                left[i] = left[i - 1] + 1;
            } else {
                left[i] = left[i - 1];
            }
        }
        set.clear();
        int[] right = new int[s.length()];
        right[right.length - 1] = 1;
        set.add(s.charAt(s.length() - 1));
        for (int i = s.length() - 2; i >= 0; --i) {
            if (set.add(s.charAt(i))) {
                right[i] = right[i + 1] + 1;
            } else {
                right[i] = right[i + 1];
            }
        }
        int res = 0;
        for (int i = 0; i < s.length() - 1; ++i) {
            if (left[i] == right[i + 1]) {
                ++res;
            }
        }
        return res;

    }

    // 89. 格雷编码 (Gray Code)
    public List<Integer> grayCode(int n) {
        List<Integer> res = new ArrayList<>();
        res.add(0);
        int head = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = res.size() - 1; j >= 0; --j) {
                res.add(head | res.get(j));
            }
            head <<= 1;
        }
        return res;

    }

    // 1238. 循环码排列 (Circular Permutation in Binary Representation)
    public List<Integer> circularPermutation(int n, int start) {
        List<Integer> list = new ArrayList<>();
        list.add(0);
        int head = 1;
        for (int i = 0; i < n; ++i) {
            for (int j = list.size() - 1; j >= 0; --j) {
                list.add(head | list.get(j));
            }
            head <<= 1;
        }
        while (list.get(0) != start) {
            list.add(list.remove(0));
        }
        return list;

    }

    // 318. 最大单词长度乘积 (Maximum Product of Word Lengths) // 剑指 Offer II 005. 单词长度的最大乘积
    public int maxProduct(String[] words) {
        int[] status = new int[words.length];
        for (int i = 0; i < words.length; ++i) {
            for (char c : words[i].toCharArray()) {
                status[i] |= 1 << (c - 'a');
            }
        }
        int res = 0;
        for (int i = 0; i < status.length; ++i) {
            for (int j = i + 1; j < status.length; ++j) {
                if ((status[i] & status[j]) == 0) {
                    res = Math.max(res, words[i].length() * words[j].length());
                }
            }
        }
        return res;

    }

    // 1734. 解码异或后的排列 (Decode XORed Permutation)
    public int[] decode(int[] encoded) {
        int total = 0;
        for (int i = 1; i <= encoded.length + 1; ++i) {
            total ^= i;
        }
        int odd = 0;
        for (int i = 1; i < encoded.length; i += 2) {
            odd ^= encoded[i];
        }
        int[] perm = new int[encoded.length + 1];
        perm[0] = odd ^ total;
        for (int i = 1; i < perm.length; ++i) {
            perm[i] = perm[i - 1] ^ encoded[i - 1];
        }
        return perm;

    }

    // 810. 黑板异或游戏 (Chalkboard XOR Game)
    public boolean xorGame(int[] nums) {
        if (nums.length % 2 == 0) {
            return true;
        }
        int xor = 0;
        for (int num : nums) {
            xor ^= num;
        }
        return xor == 0;

    }

    // 面试题 05.02. 二进制数转字符串 (Bianry Number to String LCCI)
    public String printBin(double num) {
        final int M = 1000000;
        int count = 6;
        StringBuilder res = new StringBuilder();
        res.append("0.");
        int n = (int) (num * M);
        while (count-- > 0 && n != 0) {
            n <<= 1;
            res.append(n / M);
            n %= M;
        }
        return n != 0 ? "ERROR" : res.toString();

    }

    // 784. 字母大小写全排列 (Letter Case Permutation) --回溯
    public List<String> letterCasePermutation(String s) {
        List<String> res = new ArrayList<>();
        dfs784(s.toCharArray(), 0, res);
        return res;

    }

    private void dfs784(char[] arr, int pos, List<String> res) {
        while (pos < arr.length && Character.isDigit(arr[pos])) {
            ++pos;
        }
        if (pos == arr.length) {
            res.add(String.valueOf(arr));
            return;
        }
        arr[pos] ^= 32;
        dfs784(arr, pos + 1, res);
        arr[pos] ^= 32;
        dfs784(arr, pos + 1, res);
    }

    // 784. 字母大小写全排列 (Letter Case Permutation) --二进制位图
    public List<String> letterCasePermutation2(String s) {
        List<String> res = new ArrayList<>();
        int count = 0;
        for (char c : s.toCharArray()) {
            if (Character.isLetter(c)) {
                ++count;
            }
        }

        for (int i = 0; i < (1 << count); ++i) {
            StringBuilder builder = new StringBuilder();
            int mask = i;
            for (int j = 0; j < s.length(); ++j) {
                if (Character.isLetter(s.charAt(j))) {
                    if ((mask & 1) == 0) {
                        builder.append(Character.toUpperCase(s.charAt(j)));
                    } else {
                        builder.append(Character.toLowerCase(s.charAt(j)));
                    }
                    mask >>= 1;
                } else {
                    builder.append(s.charAt(j));
                }
            }
            res.add(builder.toString());
        }
        return res;

    }

    // 784. 字母大小写全排列 (Letter Case Permutation) --bfs
    public List<String> letterCasePermutation3(String s) {
        List<String> res = new ArrayList<>();
        int n = s.length();
        Queue<StringBuilder> queue = new LinkedList<>();
        queue.offer(new StringBuilder());
        search: while (!queue.isEmpty()) {
            StringBuilder cur = queue.poll();
            if (cur.length() == n) {
                res.add(cur.toString());
                continue;
            }
            int index = cur.length();
            while (index < n) {
                if (Character.isDigit(s.charAt(index))) {
                    cur.append(s.charAt(index));
                    ++index;
                } else {
                    StringBuilder builder1 = new StringBuilder(cur);
                    builder1.append(Character.toUpperCase(s.charAt(index)));
                    queue.offer(builder1);

                    StringBuilder builder2 = new StringBuilder(cur);
                    builder2.append(Character.toLowerCase(s.charAt(index)));
                    queue.offer(builder2);

                    continue search;

                }
            }
            queue.offer(new StringBuilder(cur));

        }
        return res;

    }

    // 面试题 16.07. 最大数值 (Maximum LCCI)
    public int maximum(int a, int b) {
        long c = a;
        long d = b;
        int res = (int) ((Math.abs(c - d) + c + d) / 2);
        return res;
    }

    // 面试题 08.05. 递归乘法 (Recursive Mulitply LCCI)
    public int multiply(int A, int B) {
        if (A == 0 || B == 0) {
            return 0;
        }
        if (A < B) {
            return B + multiply(A - 1, B);
        }
        return A + multiply(A, B - 1);
    }

    // 面试题 08.05. 递归乘法 (Recursive Mulitply LCCI)
    public int multiply2(int A, int B) {
        int res = 0;
        while (B != 0) {
            if ((B & 1) != 0) {
                res += A;
            }
            B >>= 1;
            if (B != 0) {
                A += A;
            }
        }
        return res;
    }

    // 1829. 每个查询的最大异或值 (Maximum XOR for Each Query)
    public int[] getMaximumXor(int[] nums, int maximumBit) {
        int mask = (1 << maximumBit) - 1;
        int[] res = new int[nums.length];
        int xor = 0;
        for (int num : nums) {
            xor ^= num;
        }
        for (int i = nums.length - 1; i >= 0; --i) {
            res[nums.length - i - 1] = mask ^ xor;
            xor ^= nums[i];
        }
        return res;

    }

    // 1318. 或运算的最小翻转次数 (Minimum Flips to Make a OR b Equal to c)
    public int minFlips(int a, int b, int c) {
        int res = 0;
        for (int i = 0; i < 32; ++i) {
            int aBit = (a >> i) & 1;
            int bBit = (b >> i) & 1;
            int cBit = (c >> i) & 1;
            if (cBit == 0) {
                res += aBit + bBit;
            } else {
                res += aBit + bBit == 0 ? 1 : 0;
            }
        }
        return res;

    }

    // 421. 数组中两个数的最大异或值 (Maximum XOR of Two Numbers in an Array)
    // LCR 067. 数组中两个数的最大异或值
    public int findMaximumXOR(int[] nums) {
        int res = 0;
        int mask = 0;
        int max = 0;
        for (int num : nums) {
            max = Math.max(max, num);
        }
        int highestBit = 31 - Integer.numberOfLeadingZeros(max);
        Set<Integer> seen = new HashSet<>();
        for (int i = highestBit; i >= 0; --i) {
            seen.clear();
            mask |= 1 << i;
            int newRes = res | (1 << i);
            for (int num : nums) {
                num &= mask;
                if (seen.contains(num ^ newRes)) {
                    res = newRes;
                    break;
                }
                seen.add(num);
            }
        }
        return res;

    }

    // 1979. 找出数组的最大公约数 (Find Greatest Common Divisor of Array)
    public int findGCD(int[] nums) {
        int max = Arrays.stream(nums).max().getAsInt();
        int min = Arrays.stream(nums).min().getAsInt();
        return getGCD1979(max, min);

    }

    private int getGCD1979(int a, int b) {
        // gcd(a, b) = gcd(b, a mod b)
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    // 1281. 整数的各位积和之差 (Subtract the Product and Sum of Digits of an Integer)
    public int subtractProductAndSum(int n) {
        int sum = 0;
        int product = 1;
        while (n != 0) {
            int num = n % 10;
            sum += num;
            product *= num;
            n /= 10;
        }
        return product - sum;
    }

    // 1551. 使数组中所有元素相等的最小操作数 (Minimum Operations to Make Array Equal)
    public int minOperations(int n) {
        return n * n / 4;
    }

    // 面试题 05.06. 整数转换 (Convert Integer LCCI)
    public int convertInteger(int A, int B) {
        int diff = A ^ B;
        int res = 0;
        while (diff != 0) {
            diff &= diff - 1;
            ++res;
        }
        return res;

    }

    // 面试题 05.06. 整数转换 (Convert Integer LCCI)
    public int convertInteger2(int A, int B) {
        int diff = A ^ B;
        return Integer.bitCount(diff);
    }

    // 476. 数字的补数 (Number Complement)
    // 1009. 十进制整数的反码 (Complement of Base 10 Integer)
    public int bitwiseComplement(int n) {
        int u = (1 << Integer.toBinaryString(n).length()) - 1;
        return u ^ n;
    }

    // 476. 数字的补数 (Number Complement)
    // 1009. 十进制整数的反码 (Complement of Base 10 Integer)
    public int bitwiseComplement2(int n) {
        if (n == 0) {
            return 1;
        }
        int u = (1 << (32 - Integer.numberOfLeadingZeros(n))) - 1;
        return u ^ n;
    }

    // 405. 数字转换为十六进制数 (Convert a Number to Hexadecimal)
    public String toHex(int num) {
        if (num == 0) {
            return "0";
        }
        char[] table = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f' };
        String res = "";
        while (num != 0) {
            res = table[num & 0b1111] + res;
            num >>>= 4;
        }
        return res;

    }

    // 201. 数字范围按位与 (Bitwise AND of Numbers Range)
    public int rangeBitwiseAnd(int left, int right) {
        int shift = 0;
        while (left < right) {
            left >>= 1;
            right >>= 1;
            ++shift;
        }
        return left << shift;

    }

    // 201. 数字范围按位与 (Bitwise AND of Numbers Range)
    public int rangeBitwiseAnd2(int left, int right) {
        while (left < right) {
            right &= right - 1;
        }
        return right;
    }

    // 面试题 05.01. 插入 (Insert Into Bits LCCI)
    public int insertBits(int N, int M, int i, int j) {
        int w = ((1 << (j - i + 1)) - 1) << i;
        int u = (1 << (32 - Integer.numberOfLeadingZeros(N))) - 1;
        N &= w ^ u;
        return N | (M << i);
    }

    // 1461. 检查一个字符串是否包含所有长度为 K 的二进制子串 (Check If a String Contains All Binary Codes
    // of Size K)
    public boolean hasAllCodes(String s, int k) {
        if (s.length() < (1 << k) + k - 1) {
            return false;
        }
        Set<String> set = new HashSet<>();
        for (int i = k; i <= s.length(); ++i) {
            set.add(s.substring(i - k, i));
        }
        return set.size() == (1 << k);
    }

    // 1461. 检查一个字符串是否包含所有长度为 K 的二进制子串 (Check If a String Contains All Binary Codes
    // of Size K)
    public boolean hasAllCodes2(String s, int k) {
        if (s.length() < (1 << k) + k - 1) {
            return false;
        }
        int num = Integer.parseInt(s.substring(0, k), 2);
        Set<Integer> set = new HashSet<>();
        set.add(num);
        for (int i = 1; (i + k) <= s.length(); ++i) {
            num = (num - ((s.charAt(i - 1) - '0') << (k - 1))) * 2 + (s.charAt(i + k - 1) - '0');
            set.add(num);
        }
        return set.size() == (1 << k);
    }

    // 1392. 最长快乐前缀 (Longest Happy Prefix)
    public String longestPrefix(String s) {
        int left = 0;
        int right = s.length() - 1;
        String res = "";
        while (right > 0) {
            String prefix = s.substring(0, left + 1);
            String suffix = s.substring(right);
            if (prefix.equals(suffix)) {
                res = prefix;
            }
            ++left;
            --right;
        }
        return res;

    }

    // 1822. 数组元素积的符号 (Sign of the Product of an Array)
    public int arraySign(int[] nums) {
        int negative = 0;
        for (int num : nums) {
            if (num == 0) {
                return 0;
            }
            if (num < 0) {
                ++negative;
            }
        }
        return (negative & 1) == 0 ? 1 : -1;

    }

    // 1812. 判断国际象棋棋盘中一个格子的颜色 (Determine Color of a Chessboard Square)
    public boolean squareIsWhite(String coordinates) {
        int alphabet = coordinates.charAt(0) - 'a';
        int number = coordinates.charAt(1) - '0';
        return (alphabet & 1) == (number & 1);
    }

    // 1837. K 进制表示下的各位数字总和 (Sum of Digits in Base K)
    public int sumBase(int n, int k) {
        int res = 0;
        while (n != 0) {
            res += n % k;
            n /= k;
        }
        return res;

    }

    // 1561. 你可以获得的最大硬币数目 (Maximum Number of Coins You Can Get)
    public int maxCoins(int[] piles) {
        Arrays.sort(piles);
        int res = 0;
        for (int i = piles.length / 3; i < piles.length; i += 2) {
            res += piles[i];
        }
        return res;

    }

    // 877. 石子游戏 (Stone Game) --[还需要掌握动态规划算法]
    public boolean stoneGame(int[] piles) {
        return true;

    }

    // 1716. 计算力扣银行的钱 (Calculate Money in Leetcode Bank)
    public int totalMoney(int n) {
        int firstWeek = (1 + 7) * 7 / 2;
        int weeks = n / 7;
        int lastWeek = firstWeek + (weeks - 1) * 7;
        int left = n % 7;
        int sum = (firstWeek + lastWeek) * weeks / 2 + (2 * weeks + 1 + left) * left / 2;
        return sum;

    }

    // 908. 最小差值 I (Smallest Range I)
    public int smallestRangeI(int[] nums, int k) {
        int min = nums[0];
        int max = nums[0];
        for (int num : nums) {
            min = Math.min(min, num);
            max = Math.max(max, num);
        }
        return Math.max(0, max - min - 2 * k);

    }

    // 1925. 统计平方和三元组的数目 (Count Square Sum Triples)
    public int countTriples(int n) {
        int res = 0;
        for (int a = 1; a <= n; ++a) {
            for (int b = 1; b <= n; ++b) {
                int c = (int) Math.sqrt(a * a + b * b);
                if (c <= n && a * a + b * b == c * c) {
                    ++res;
                }
            }
        }
        return res;

    }

    // 789. 逃脱阻碍者 (Escape The Ghosts)
    public boolean escapeGhosts(int[][] ghosts, int[] target) {
        int[] source = { 0, 0 };
        int minDistance = getManhattanDistance789(source, target);
        for (int[] ghost : ghosts) {
            if (minDistance >= getManhattanDistance789(ghost, target)) {
                return false;
            }
        }
        return true;

    }

    private int getManhattanDistance789(int[] point1, int[] point2) {
        return Math.abs(point1[0] - point2[0]) + Math.abs(point1[1] - point2[1]);
    }

    // LCP 11. 期望个数统计
    public int expectNumber(int[] scores) {
        Set<Integer> set = new HashSet<>();
        for (int score : scores) {
            set.add(score);
        }
        return set.size();

    }

    // 973. 最接近原点的 K 个点 (K Closest Points to Origin)
    public int[][] kClosest(int[][] points, int k) {
        Arrays.sort(points, (o1, o2) -> {
            int distance1 = o1[0] * o1[0] + o1[1] * o1[1];
            int distance2 = o2[0] * o2[0] + o2[1] * o2[1];
            return distance1 - distance2;
        });
        return Arrays.copyOfRange(points, 0, k);

    }

    // 1832. 判断句子是否为全字母句 (Check if the Sentence Is Pangram)
    public boolean checkIfPangram(String sentence) {
        int mask = 0;
        for (char c : sentence.toCharArray()) {
            mask |= 1 << (c - 'a');
            if (mask == ((1 << 26) - 1)) {
                return true;
            }
        }
        return false;

    }

    // 50. Pow(x, n) (Pow(x, n)) --快速幂 迭代
    // 剑指 Offer 16. 数值的整数次方
    public double myPow(double x, int n) {
        long N = n;
        return n > 0 ? getQuickMultiplyIteration(x, N) : 1.0 / getQuickMultiplyIteration(x, -N);

    }

    private double getQuickMultiplyIteration(double x, long n) {
        double res = 1.0;
        while (n > 0) {
            if ((n & 1) != 0) {
                res *= x;
            }
            x *= x;
            n >>= 1;
        }
        return res;
    }

    // 50. Pow(x, n) (Pow(x, n)) --快速幂 递归
    // 剑指 Offer 16. 数值的整数次方
    public double myPow2(double x, int n) {
        long N = n;
        return n > 0 ? getQuickMultiplyRecursion(x, N) : 1.0 / getQuickMultiplyRecursion(x, -N);

    }

    private double getQuickMultiplyRecursion(double x, long n) {
        if (n == 0) {
            return 1.0d;
        }
        double y = getQuickMultiplyRecursion(x, n / 2);
        return n % 2 == 0 ? y * y : y * y * x;
    }

    // 172. 阶乘后的零 (Factorial Trailing Zeroes)
    // 面试题 16.05. 阶乘尾数 (Factorial Zeros LCCI)
    /**
     * 判断有多少个5的因子
     */
    public int trailingZeroes(int n) {
        int count = 0;
        for (int i = 5; i <= n; i += 5) {
            int N = i;
            while (N > 0) {
                if (N % 5 == 0) {
                    ++count;
                    N /= 5;
                } else {
                    break;
                }
            }
        }
        return count;
    }

    // 172. 阶乘后的零 (Factorial Trailing Zeroes)
    // 面试题 16.05. 阶乘尾数 (Factorial Zeros LCCI)
    /**
     * 判断有多少个5的因子 每隔 5 个数，出现一个 5，每隔 25 个数，出现 2 个 5，每隔 125 个数，出现 3 个 5... 以此类推
     */
    public int trailingZeroes2(int n) {
        int res = 0;
        while (n > 0) {
            res += n / 5;
            n /= 5;
        }
        return res;

    }

    // 60. 排列序列 (Permutation Sequence)
    public String getPermutation(int n, int k) {
        char[] permutations = new char[n];
        for (int i = 0; i < n; ++i) {
            permutations[i] = (char) (i + 1 + '0');
        }
        for (int i = 0; i < k - 1; ++i) {
            int j = permutations.length - 1;
            while (j > 0) {
                if (permutations[j - 1] < permutations[j]) {
                    break;
                }
                --j;
            }
            --j;
            int l = permutations.length - 1;
            while (j < l) {
                if (permutations[j] < permutations[l]) {
                    char temp = permutations[j];
                    permutations[j] = permutations[l];
                    permutations[l] = temp;
                    break;
                }
                --l;
            }
            Arrays.sort(permutations, j + 1, permutations.length);
        }
        return String.valueOf(permutations);

    }

    // 168. Excel表列名称 (Excel Sheet Column Title)
    public String convertToTitle(int columnNumber) {
        StringBuilder res = new StringBuilder();
        while (columnNumber > 0) {
            --columnNumber;
            char temp = (char) (columnNumber % 26 + 'A');
            res.append(temp);
            columnNumber /= 26;
        }
        return res.reverse().toString();

    }

    // 1970.你能穿过矩阵的最后一天 (Last Day Where You Can Still Cross) --bfs
    public int latestDayToCross(int row, int col, int[][] cells) {
        int left = 0;
        int right = cells.length - 1;
        int res = -1;
        while (left <= right) {
            int mid = left + ((right - left) >>> 1);
            if (check1970(row, col, cells, mid)) {
                res = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return res;

    }

    private boolean check1970(int row, int col, int[][] cells, int mid) {
        int[][] directions = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
        int[][] arr = new int[row][col];
        for (int i = 0; i < row; ++i) {
            Arrays.fill(arr[i], 1);
        }
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < mid; ++i) {
            int x = cells[i][0] - 1;
            int y = cells[i][1] - 1;
            arr[x][y] = 0;
        }
        for (int j = 0; j < col; ++j) {
            if (arr[0][j] == 1) {
                queue.offer(new int[] { 0, j });
                arr[0][j] = 0;
            }

        }
        while (!queue.isEmpty()) {
            int[] cur = queue.poll();
            if (cur[0] == row - 1) {
                return true;
            }
            for (int[] direction : directions) {
                int nx = cur[0] + direction[0];
                int ny = cur[1] + direction[1];
                if (nx >= 0 && nx < row && ny >= 0 && ny < col && arr[nx][ny] == 1) {
                    arr[nx][ny] = 0;
                    queue.offer(new int[] { nx, ny });
                }
            }
        }
        return false;
    }

    // 1970. 你能穿过矩阵的最后一天 (Last Day Where You Can Still Cross) --并查集
    public int latestDayToCross2(int row, int col, int[][] cells) {
        int n = cells.length;
        Union1970 uf = new Union1970(n + 2);
        for (int i = 0; i < col; ++i) {
            uf.union(i, n);
            uf.union(n - i - 1, n + 1);
        }
        int[][] grid = new int[row][col];
        int[][] dirs = { { -1, 0 }, { 1, 0 }, { 0, -1 }, { 0, 1 } };
        for (int i = n - 1; i >= 0; --i) {
            int r = cells[i][0] - 1;
            int c = cells[i][1] - 1;
            grid[r][c] = 1;
            for (int[] d : dirs) {
                int nr = r + d[0];
                int nc = c + d[1];
                if (nr >= 0 && nr < row && nc >= 0 && nc < col && grid[nr][nc] == 1) {
                    uf.union(r * col + c, nr * col + nc);
                }
            }
            if (uf.find(n) == uf.find(n + 1)) {
                return i;
            }
        }
        return -1;

    }

    public class Union1970 {
        private int[] parent;
        private int[] rank;

        public Union1970(int n) {
            this.parent = new int[n];
            this.rank = new int[n];
            for (int i = 0; i < n; ++i) {
                parent[i] = i;
                rank[i] = 1;
            }
        }

        public int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }

        public void union(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX != rootY) {
                if (rank[rootX] > rank[rootY]) {
                    parent[rootY] = rootX;
                } else if (rank[rootX] < rank[rootY]) {
                    parent[rootX] = rootY;
                } else {
                    parent[rootY] = rootX;
                    rank[rootX]++;
                }
            }
        }
    }

    // 241. 为运算表达式设计优先级 (Different Ways to Add Parentheses)
    public List<Integer> diffWaysToCompute(String expression) {
        List<Integer> res = new ArrayList<>();
        int start = 0;
        while (start < expression.length()) {
            if (!Character.isDigit(expression.charAt(start))) {
                break;
            }
            ++start;
        }
        if (start == expression.length()) {
            res.add(Integer.parseInt(expression));
        }
        for (int i = start; i < expression.length(); ++i) {
            if (!Character.isDigit(expression.charAt(i))) {
                char op = expression.charAt(i);
                List<Integer> left = diffWaysToCompute(expression.substring(0, i));
                List<Integer> right = diffWaysToCompute(expression.substring(i + 1, expression.length()));
                for (int x : left) {
                    for (int y : right) {
                        if (op == '+') {
                            res.add(x + y);
                        } else if (op == '-') {
                            res.add(x - y);
                        } else {
                            res.add(x * y);
                        }
                    }
                }
            }
        }
        return res;
    }

    // 204. 计数质数 (Count Primes)
    public int countPrimes(int n) {
        boolean[] isPrime = new boolean[n];
        Arrays.fill(isPrime, true);
        for (int i = 2; i * i < n; ++i) {
            if (isPrime[i]) {
                for (int j = i * i; j < n; j += i) {
                    isPrime[j] = false;
                }
            }
        }
        int res = 0;
        for (int i = 2; i < n; ++i) {
            if (isPrime[i]) {
                ++res;
            }
        }
        return res;

    }

    // 1998. 数组的最大公因数排序 (GCD Sort of an Array)
    public boolean gcdSort(int[] nums) {
        Union1998 union = new Union1998(100005);
        for (int num : nums) {
            int c = num;
            for (int i = 2; i <= num / i; ++i) {
                boolean flag = false;
                while (num % i == 0) {
                    num /= i;
                    flag = true;
                }
                if (flag) {
                    union.union(c, i);
                }
            }
            if (num > 1) {
                union.union(c, num);
            }
        }
        int[] sort = nums.clone();
        Arrays.sort(sort);
        for (int i = 0; i < nums.length; ++i) {
            if (nums[i] != sort[i] && !union.isConnected(nums[i], sort[i])) {
                return false;
            }
        }
        return true;

    }

    public class Union1998 {
        private int[] rank;
        private int[] parent;

        public Union1998(int n) {
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
            } else {
                parent[root2] = root1;
                if (rank[root1] == rank[root2]) {
                    ++rank[root1];
                }
            }
        }
    }

    // 504. 七进制数 (Base 7)
    public String convertToBase7(int num) {
        if (num == 0) {
            return "0";
        }
        StringBuilder res = new StringBuilder();
        int sign = num < 0 ? -1 : 1;
        num *= sign;
        while (num != 0) {
            res.append(num % 7);
            num /= 7;
        }
        if (sign < 0) {
            res.append('-');
        }
        return res.reverse().toString();

    }

    // 598. 范围求和 II (Range Addition II)
    public int maxCount(int m, int n, int[][] ops) {
        int minM = m;
        int minN = n;
        for (int[] op : ops) {
            minM = Math.min(minM, op[0]);
            minN = Math.min(minN, op[1]);
        }
        return minM * minN;

    }

    // 507. 完美数 (Perfect Number)
    public boolean checkPerfectNumber(int num) {
        if (num == 1) {
            return false;
        }
        int sum = 1;
        for (int i = 2; i * i <= num; ++i) {
            if (num % i == 0) {
                sum += i;
                if (i * i != num) {
                    sum += num / i;
                }
            }
        }
        return num == sum;

    }

    // 486. 预测赢家 (Predict the Winner) 二维dp
    public boolean PredictTheWinner(int[] nums) {
        int n = nums.length;
        int[][] dp = new int[n][n];
        for (int i = 0; i < n; ++i) {
            dp[i][i] = nums[i];
        }
        for (int i = n - 2; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                dp[i][j] = Math.max(nums[i] - dp[i + 1][j], nums[j] - dp[i][j - 1]);
            }
        }

        return dp[0][n - 1] >= 0;

    }

    // 486. 预测赢家 (Predict the Winner) 一维dp
    public boolean PredictTheWinner2(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        for (int i = 0; i < n; ++i) {
            dp[i] = nums[i];
        }
        for (int i = n - 2; i >= 0; --i) {
            for (int j = i + 1; j < n; ++j) {
                dp[j] = Math.max(nums[i] - dp[j], nums[j] - dp[j - 1]);
            }
        }

        return dp[n - 1] >= 0;

    }

    // 788. 旋转数字 (Rotated Digits) --数位dp
    private char[] chars788;
    private int m788;

    public int rotatedDigits2(int n) {
        chars788 = String.valueOf(n).toCharArray();
        this.m788 = chars788.length;
        Set<Integer> s1 = new HashSet<>(List.of(0, 1, 2, 5, 6, 8, 9));
        Set<Integer> s2 = new HashSet<>(List.of(0, 1, 8));
        return check(s1) - check(s2);
    }

    private Set<Integer> set788;
    private int[] memo788;

    private int check(Set<Integer> set) {
        this.set788 = set;
        this.memo788 = new int[m788];
        Arrays.fill(memo788, -1);
        return dfs788(0, true);
    }

    private int dfs788(int i, boolean isLimit) {
        if (i == m788) {
            return 1;
        }
        if (!isLimit && memo788[i] != -1) {
            return memo788[i];
        }
        int res = 0;
        int up = isLimit ? chars788[i] - '0' : 9;
        for (int d = 0; d <= up; ++d) {
            if (set788.contains(d)) {
                res += dfs788(i + 1, isLimit && d == up);
            }
        }
        if (!isLimit) {
            memo788[i] = res;
        }
        return res;
    }

    // 858. 镜面反射 (Mirror Reflection)
    public int mirrorReflection(int p, int q) {
        int gcd = gcd858(p, q);
        p /= gcd;
        p %= 2;
        q /= gcd;
        q %= 2;
        if (p == 1 && q == 1) {
            return 1;
        }
        return (p & 1) == 1 ? 0 : 2;
    }

    private int gcd858(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    // 423. 从英文中重建数字 (Reconstruct Original Digits from English)
    public String originalDigits(String s) {
        int[] alphabet = new int[26];
        for (char c : s.toCharArray()) {
            ++alphabet[c - 'a'];
        }
        int[] count = new int[10];
        // zero 0
        // z
        count[0] = operateDigits(alphabet, 'z', new char[] { 'z', 'e', 'r', 'o' });
        // eight 8
        // g
        count[8] = operateDigits(alphabet, 'g', new char[] { 'e', 'i', 'g', 'h', 't' });
        // six 6
        // x
        count[6] = operateDigits(alphabet, 'x', new char[] { 's', 'i', 'x' });
        // seven 7
        // s
        count[7] = operateDigits(alphabet, 's', new char[] { 's', 'e', 'v', 'e', 'n' });
        // five 5
        // v
        count[5] = operateDigits(alphabet, 'v', new char[] { 'f', 'i', 'v', 'e' });
        // four 4
        // u
        count[4] = operateDigits(alphabet, 'u', new char[] { 'f', 'o', 'u', 'r' });
        // three 3
        // r
        count[3] = operateDigits(alphabet, 'r', new char[] { 't', 'h', 'r', 'e', 'e' });
        // two 2
        // w
        count[2] = operateDigits(alphabet, 'w', new char[] { 't', 'w', 'o' });
        // one 1
        // o
        count[1] = operateDigits(alphabet, 'o', new char[] { 'o', 'n', 'e' });
        // nine 9
        // i
        count[9] = operateDigits(alphabet, 'i', new char[] { 'n', 'i', 'n', 'e' });
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < count.length; ++i) {
            for (int j = 0; j < count[i]; ++j) {
                res.append(i);
            }
        }
        return res.toString();
    }

    private int operateDigits(int[] alphabet, char c, char[] digits) {
        int count = alphabet[c - 'a'];
        if (count == 0) {
            return count;
        }
        for (char digit : digits) {
            alphabet[digit - 'a'] -= count;
        }
        return count;
    }

    // 1317. 将整数转换为两个无零整数的和 (Convert Integer to the Sum of Two No-Zero Integers)
    public int[] getNoZeroIntegers(int n) {
        for (int i = 1; i < n; ++i) {
            if (check1317(i) && check1317(n - i)) {
                return new int[] { i, n - i };
            }
        }
        return null;

    }

    private boolean check1317(int x) {
        while (x > 0) {
            if (x % 10 == 0) {
                return false;
            }
            x /= 10;
        }
        return true;
    }

    // 1523. 在区间范围内统计奇数数目 (Count Odd Numbers in an Interval Range)
    public int countOdds(int low, int high) {
        return (high - low + 1) / 2 + ((high & 1) + (low & 1)) / 2;
    }

    // 1523. 在区间范围内统计奇数数目 (Count Odd Numbers in an Interval Range)
    public int countOdds2(int low, int high) {
        return pre1523(high) - pre1523(low - 1);
    }

    // 从 0--num 奇数的个数
    private int pre1523(int num) {
        return (num + 1) >> 1;
    }

    // 1952. 三除数 (Three Divisors)
    public boolean isThree(int n) {
        int count = 0;
        for (int i = 1; i * i <= n; ++i) {
            if (n % i == 0) {
                ++count;
                if (i * i != n) {
                    ++count;
                }
                if (count > 3) {
                    return false;
                }
            }
        }
        return count == 3;

    }

    // 1071. 字符串的最大公因子 (Greatest Common Divisor of Strings)
    public String gcdOfStrings(String str1, String str2) {
        if (!(str1 + str2).equals(str2 + str1)) {
            return "";
        }
        return str1.substring(0, gcd1071(str1.length(), str2.length()));

    }

    private int gcd1071(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    // 1154. 一年中的第几天 (Day of the Year)
    public int dayOfYear(String date) {
        int y = Integer.parseInt(date.substring(0, 4));
        int m = Integer.parseInt(date.substring(5, 7));
        int d = Integer.parseInt(date.substring(8));
        boolean leapYear = y % 4 == 0 && y % 100 != 0 || y % 400 == 0;
        int[] mon = { 31, 28 + (leapYear ? 1 : 0), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        int res = 0;
        for (int i = 1; i < m; ++i) {
            res += mon[i - 1];
        }
        res += d;
        return res;
    }

    // 是否为闰年
    private boolean isLeapYear(int year) {
        return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    }

    // 1360. 日期之间隔几天 (Number of Days Between Two Dates)
    public int daysBetweenDates(String date1, String date2) {
        if (date1.compareTo(date2) > 0) {
            String temp = date1;
            date1 = date2;
            date2 = temp;
        }
        int year1 = Integer.parseInt(date1.substring(0, 4));
        int month1 = Integer.parseInt(date1.substring(5, 7));
        int day1 = Integer.parseInt(date1.substring(8, 10));

        int year2 = Integer.parseInt(date2.substring(0, 4));
        int month2 = Integer.parseInt(date2.substring(5, 7));
        int day2 = Integer.parseInt(date2.substring(8, 10));
        int[] daysOfMonth1 = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        int[] daysOfMonth2 = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        if (isLeapYear(year1)) {
            daysOfMonth1[1] = 29;
        }
        if (isLeapYear(year2)) {
            daysOfMonth2[1] = 29;
        }

        int res = 0;

        for (int year = year1 + 1; year <= year2 - 1; ++year) {
            if (isLeapYear(year)) {
                res += 366;
            } else {
                res += 365;
            }
        }
        // R表示 year2的1月1号 到date2 的天数
        int R = 0;
        // L表示 year1的1月1号 到date1 的天数
        int L = 0;
        for (int month = 0; month <= month1 - 2; ++month) {
            L += daysOfMonth1[month];
        }
        L += day1;

        for (int month = 0; month <= month2 - 2; ++month) {
            R += daysOfMonth2[month];
        }
        R += day2;
        if (year1 == year2) {
            return res + R - L;
        } else if (isLeapYear(year1)) {
            return res + 366 - L + R;
        }
        return res + 365 - L + R;

    }

    // 264. 丑数 II (Ugly Number II)
    // 剑指 Offer 49. 丑数
    public int nthUglyNumber(int n) {
        int[] dp = new int[n + 1];
        dp[1] = 1;
        int p2 = 1;
        int p3 = 1;
        int p5 = 1;

        for (int i = 2; i <= n; ++i) {
            int num2 = dp[p2] * 2;
            int num3 = dp[p3] * 3;
            int num5 = dp[p5] * 5;
            dp[i] = Math.min(Math.min(num2, num3), num5);
            if (dp[i] == num2) {
                ++p2;
            }
            if (dp[i] == num3) {
                ++p3;
            }
            if (dp[i] == num5) {
                ++p5;
            }
        }
        return dp[n];

    }

    // 面试题 17.09. 第 k 个数 (Get Kth Magic Number LCCI) --优先队列
    public int getKthMagicNumber(int k) {
        int[] factors = { 3, 5, 7 };
        Queue<Long> queue = new PriorityQueue<>();
        Set<Long> seen = new HashSet<>();
        queue.offer(1L);
        seen.add(1L);
        int res = 0;
        while (k-- > 0) {
            long poll = queue.poll();
            res = (int) poll;
            for (int factor : factors) {
                long next = factor * poll;
                if (seen.add(next)) {
                    queue.offer(next);
                }
            }
        }
        return res;
    }

    // 面试题 17.09. 第 k 个数 (Get Kth Magic Number LCCI)
    public int getKthMagicNumber2(int k) {
        int[] dp = new int[k + 1];
        int p3 = 1;
        int p5 = 1;
        int p7 = 1;
        dp[1] = 1;
        for (int i = 2; i < dp.length; ++i) {
            int num3 = dp[p3] * 3;
            int num5 = dp[p5] * 5;
            int num7 = dp[p7] * 7;
            dp[i] = Math.min(Math.min(num3, num5), num7);
            if (dp[i] == num3) {
                ++p3;
            }
            if (dp[i] == num5) {
                ++p5;
            }
            if (dp[i] == num7) {
                ++p7;
            }
        }
        return dp[k];
    }

    // 剑指 Offer 10- II. 青蛙跳台阶问题
    public int numWays(int n) {
        final int MOD = 1000000007;
        if (n <= 1) {
            return 1;
        }
        int first = 1;
        int second = 2;
        for (int i = 3; i <= n; ++i) {
            int temp = second;
            second += first;
            first = temp;
            first %= MOD;
            second %= MOD;
        }
        return second;
    }

    // 面试题 08.01. 三步问题 (Three Steps Problem LCCI)
    public int waysToStep(int n) {
        if (n == 1) {
            return 1;
        }
        if (n == 2) {
            return 2;
        }
        if (n == 3) {
            return 4;
        }
        final int MOD = 1000000007;
        int first = 1;
        int second = 2;
        int third = 4;
        for (int i = 4; i <= n; ++i) {
            int temp = ((first + second) % MOD + third) % MOD;
            first = second;
            second = third;
            third = temp;
        }
        return third;

    }

    // 70. 爬楼梯 (Climbing Stairs) --记忆化搜索
    private int[] memo70;

    public int climbStairs(int n) {
        memo70 = new int[n + 1];
        return dfs70(n);

    }

    private int dfs70(int n) {
        if (n <= 1) {
            return n;
        }
        if (memo70[n] != 0) {
            return memo70[n];
        }
        return memo70[n] = dfs70(n - 1) + dfs70(n - 2);

    }

    // 70. 爬楼梯 (Climbing Stairs)
    public int climbStairs2(int n) {
        if (n == 1) {
            return n;
        }
        int first = 1;
        int second = 2;
        for (int i = 3; i <= n; ++i) {
            int temp = first + second;
            first = second;
            second = temp;
        }
        return second;
    }

    // LCS 01. 下载插件
    public int leastMinutes(int n) {
        int res = 0;
        int speed = 1;
        while (speed < n) {
            speed <<= 1;
            ++res;
        }
        return res + 1;

    }

    // LCS 01. 下载插件
    public int leastMinutes2(int n) {
        return (int) (Math.ceil(Math.log(n) / Math.log(2)) + 1);

    }

    // 1137. 第 N 个泰波那契数 (N-th Tribonacci Number)
    public int tribonacci(int n) {
        if (n <= 1) {
            return n;
        }
        int first = 0;
        int second = 1;
        int third = 1;
        for (int i = 3; i <= n; ++i) {
            int temp = first + second + third;
            first = second;
            second = third;
            third = temp;
        }
        return third;

    }

    // 1844. 将所有数字用字符替换 (Replace All Digits with Characters)
    public String replaceDigits(String s) {
        char[] res = s.toCharArray();
        for (int i = 1; i < s.length(); i += 2) {
            res[i] = (char) (res[i - 1] + res[i] - '0');
        }
        return new String(res);

    }

    // 1528. 重新排列字符串 (Shuffle String)
    public String restoreString2(String s, int[] indices) {
        char[] arr = new char[s.length()];
        for (int i = 0; i < indices.length; ++i) {
            arr[indices[i]] = s.charAt(i);
        }
        return String.valueOf(arr);

    }

    // 1017. 负二进制转换 (Convert to Base -2)
    public String baseNeg2(int n) {
        int upperLimit = 1;
        while (0 < upperLimit && upperLimit < n) {
            upperLimit <<= 2;
            upperLimit |= 1;
        }
        return Integer.toBinaryString(upperLimit ^ upperLimit - n);

    }

    // 1344. 时钟指针的夹角 (Angle Between Hands of a Clock)
    public double angleClock(int hour, int minutes) {
        // 0-minutes的度数 360 * minutes / 60
        double degreeMinute = minutes * 6;
        // 0-hour的度数 360 * (minutes * 5 / 60) / 60 + 360 * hour / 12;
        double degreeHour = minutes / 2d + hour * 30d;
        double abs = Math.abs(degreeMinute - degreeHour);
        return Math.min(abs, 360 - abs);

    }

    // 462. 最少移动次数使数组元素相等 II 找中位数
    public int minMoves2(int[] nums) {
        Arrays.sort(nums);
        int res = 0;
        for (int num : nums) {
            res += Math.abs(nums[nums.length / 2] - num);
        }
        return res;

    }

    // 1247. 交换字符使得字符串相同 (Minimum Swaps to Make Strings Equal)
    public int minimumSwap(String s1, String s2) {
        int[] counts = new int[2];
        int n = s1.length();
        for (int i = 0; i < n; ++i) {
            if (s1.charAt(i) != s2.charAt(i)) {
                ++counts[s1.charAt(i) % 2];
            }
        }
        int d = counts[0] + counts[1];
        return d % 2 != 0 ? -1 : d / 2 + counts[0] % 2;
    }

    // 447. 回旋镖的数量 (Number of Boomerangs)
    public int numberOfBoomerangs(int[][] points) {
        int res = 0;
        for (int[] point : points) {
            Map<Integer, Integer> map = new HashMap<>();
            for (int[] point2 : points) {
                int distance = (point[0] - point2[0]) * (point[0] - point2[0])
                        + (point[1] - point2[1]) * (point[1] - point2[1]);
                map.put(distance, map.getOrDefault(distance, 0) + 1);
            }
            for (int count : map.values()) {
                res += count * (count - 1);
            }
        }
        return res;

    }

    // 1436. 旅行终点站 (Destination City)
    public String destCity(List<List<String>> paths) {
        Map<String, String> map = new HashMap<>();
        for (List<String> p : paths) {
            map.put(p.get(0), p.get(1));
        }
        String res = paths.get(0).get(0);
        while (map.get(res) != null) {
            res = map.get(res);
        }
        return res;

    }

    // 1436. 旅行终点站 (Destination City)
    public String destCity2(List<List<String>> paths) {
        Map<String, String> map = new HashMap<>();
        for (List<String> path : paths) {
            map.put(path.get(0), path.get(1));
        }
        String res = paths.get(0).get(0);
        while (map.containsKey(res)) {
            String val = map.get(res);
            res = val;
        }
        return res;

    }

    // 2000. 反转单词前缀 (Reverse Prefix of Word)
    public String reversePrefix(String word, char ch) {
        int index = word.indexOf(ch);
        if (index == -1) {
            return word;
        }
        StringBuilder builder = new StringBuilder(word.substring(0, index + 1));
        return builder.reverse().append(word.substring(index + 1)).toString();

    }

    // 1967. 作为子字符串出现在单词中的字符串数目
    // 1967. Number of Strings That Appear as Substrings in Word
    public int numOfStrings(String[] patterns, String word) {
        int res = 0;
        for (String pattern : patterns) {
            if (word.indexOf(pattern) != -1) {
                ++res;
            }
        }
        return res;

    }

    // 781. 森林中的兔子 (Rabbits in Forest)
    public int numRabbits(int[] answers) {
        Map<Integer, Integer> map = new HashMap<>();
        int res = 0;
        for (int x : answers) {
            map.merge(x, 1, Integer::sum);
            if (map.get(x) - 1 == x) {
                res += map.remove(x);
            }
        }
        for (Map.Entry<Integer, Integer> entry : map.entrySet()) {
            res += entry.getKey() + 1;
        }
        return res;
    }

    // 1447. 最简分数 (Simplified Fractions)
    public List<String> simplifiedFractions(int n) {
        List<String> res = new ArrayList<>();
        for (int b = 2; b <= n; ++b) {
            for (int a = 1; a < b; ++a) {
                if (getGCD1447(a, b) == 1) {
                    res.add(a + "/" + b);
                }
            }
        }
        return res;

    }

    private int getGCD1447(int a, int b) {
        while (b != 0) {
            int temp = b;
            b = a % b;
            a = temp;
        }
        return a;
    }

    // 650. 只有两个键的键盘 (2 Keys Keyboard) dp
    public int minSteps(int n) {
        int[] dp = new int[n + 1];
        for (int i = 2; i < dp.length; ++i) {
            dp[i] = Integer.MAX_VALUE;
            for (int j = 1; j * j <= i; ++j) {
                if (i % j == 0) {
                    dp[i] = Math.min(dp[i], dp[j] + i / j);
                    dp[i] = Math.min(dp[i], dp[i / j] + j);
                }
            }
        }
        return dp[n];

    }

    // 650. 只有两个键的键盘 (2 Keys Keyboard) 质因数分解
    public int minSteps2(int n) {
        int res = 0;
        for (int i = 2; i * i <= n; ++i) {
            while (n % i == 0) {
                res += i;
                n /= i;
            }
        }
        if (n > 1) {
            res += n;
        }
        return res;

    }

    // 738. 单调递增的数字 (Monotone Increasing Digits)
    public int monotoneIncreasingDigits(int n) {
        char[] chars = String.valueOf(n).toCharArray();
        int i = chars.length - 2;
        while (i >= 0) {
            if (chars[i] > chars[i + 1]) {
                chars[i] -= 1;
                Arrays.fill(chars, i + 1, chars.length, '9');
            }
            --i;
        }
        return Integer.parseInt(String.valueOf(chars));

    }

    // 779. 第K个语法符号 (K-th Symbol in Grammar)
    public int kthGrammar(int n, int k) {
        if (n == 1) {
            return 0;
        }
        int length = (int) Math.pow(2, n - 1);
        if (k > length / 2) {
            int val = kthGrammar(n - 1, k - length / 2);
            return val == 0 ? 1 : 0;
        } else {
            return kthGrammar(n - 1, k);
        }
    }

    // 779. 第K个语法符号 (K-th Symbol in Grammar)
    public int kthGrammar2(int n, int k) {
        return n == 1 ? 0 : (1 - k % 2) ^ kthGrammar2(n - 1, (k + 1) / 2);

    }

    // 2011. 执行操作后的变量值 (Final Value of Variable After Performing Operations)
    public int finalValueAfterOperations(String[] operations) {
        int res = 0;
        for (String operation : operations) {
            res += operation.charAt(1) == '+' ? 1 : -1;
        }
        return res;

    }

    // 1929. 数组串联 (Concatenation of Array)
    public int[] getConcatenation(int[] nums) {
        int[] res = new int[nums.length * 2];
        System.arraycopy(nums, 0, res, 0, nums.length);
        System.arraycopy(nums, 0, res, nums.length, nums.length);
        return res;

    }

    // 2006. 差的绝对值为 K 的数对数目 (Count Number of Pairs With Absolute Difference K)
    public int countKDifference(int[] nums, int k) {
        int[] counts = new int[101];
        for (int num : nums) {
            ++counts[num];
        }
        int res = 0;
        int left = 1;
        int right = 2;
        while (right < counts.length) {
            if (right - left < k) {
                ++right;
            } else if (right - left > k) {
                ++left;
            } else {
                res += counts[right] * counts[left];
                ++left;
                ++right;
            }
        }
        return res;

    }

    // 807. 保持城市天际线 (Max Increase to Keep City Skyline)
    public int maxIncreaseKeepingSkyline(int[][] grid) {
        int n = grid.length;
        int[] rowMax = new int[n];
        int[] colMax = new int[n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                rowMax[i] = Math.max(rowMax[i], grid[i][j]);
                colMax[j] = Math.max(colMax[j], grid[i][j]);
            }
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                res += Math.min(rowMax[i], colMax[j]) - grid[i][j];
            }
        }
        return res;

    }

    // 1913. 两个数对之间的最大乘积差 (Maximum Product Difference Between Two Pairs)
    public int maxProductDifference(int[] nums) {
        int min1 = Integer.MAX_VALUE;
        int min2 = Integer.MAX_VALUE;
        int max1 = 0;
        int max2 = 0;
        for (int num : nums) {
            if (num >= max1) {
                max2 = max1;
                max1 = num;
            } else if (num > max2) {
                max2 = num;
            }

            if (num <= min1) {
                min2 = min1;
                min1 = num;
            } else if (num < min2) {
                min2 = num;
            }
        }
        return max1 * max2 - min1 * min2;

    }

    // 1637.两点之间不包含任何点的最宽垂直面积 (Widest Vertical Area Between Two Points Containing No
    // Points)
    public int maxWidthOfVerticalArea(int[][] points) {
        Arrays.sort(points, (o1, o2) -> o1[0] - o2[0]);
        int res = 0;
        for (int i = 1; i < points.length; ++i) {
            res = Math.max(res, points[i][0] - points[i - 1][0]);
        }
        return res;

    }

    // 1725. 可以形成最大正方形的矩形数目 (Number Of Rectangles That Can Form The Largest Square)
    public int countGoodRectangles(int[][] rectangles) {
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for (int[] rectangle : rectangles) {
            int side = Math.min(rectangle[0], rectangle[1]);
            map.put(side, map.getOrDefault(side, 0) + 1);
        }
        return map.lastEntry().getValue();

    }

    // 1725. 可以形成最大正方形的矩形数目 (Number Of Rectangles That Can Form The Largest Square)
    public int countGoodRectangles2(int[][] rectangles) {
        int maxSide = 0;
        int res = 0;
        for (int[] rectangle : rectangles) {
            int side = Math.min(rectangle[0], rectangle[1]);
            if (side > maxSide) {
                maxSide = side;
                res = 1;
            } else if (side == maxSide) {
                ++res;
            }
        }
        return res;

    }

    // 1725. 可以形成最大正方形的矩形数目 (Number Of Rectangles That Can Form The Largest Square)
    public int countGoodRectangles3(int[][] rectangles) {
        Arrays.sort(rectangles, new Comparator<int[]>() {

            @Override
            public int compare(int[] o1, int[] o2) {
                int side1 = Math.min(o1[0], o1[1]);
                int side2 = Math.min(o2[0], o2[1]);
                return side2 - side1;
            }
        });
        int max = Math.min(rectangles[0][0], rectangles[0][1]);
        int res = 1;
        for (int i = 1; i < rectangles.length; ++i) {
            int cur = Math.min(rectangles[i][0], rectangles[i][1]);
            if (cur != max) {
                break;
            }
            ++res;
        }
        return res;

    }

    // 1630. 等差子数组 (Arithmetic Subarrays)
    public List<Boolean> checkArithmeticSubarrays(int[] nums, int[] l, int[] r) {
        List<Boolean> res = new ArrayList<>();
        int m = l.length;
        search: for (int i = 0; i < m; ++i) {
            int left = l[i];
            int right = r[i];
            int min = nums[left];
            int max = nums[right];
            for (int j = left; j <= right; ++j) {
                min = Math.min(min, nums[j]);
                max = Math.max(max, nums[j]);
            }
            if (max == min) {
                res.add(true);
                continue;
            }
            if ((max - min) % (right - left) != 0) {
                res.add(false);
                continue;
            }
            int d = (max - min) / (right - left);
            boolean[] seen = new boolean[right - left + 1];
            for (int j = left; j <= right; ++j) {
                if ((nums[j] - min) % d != 0) {
                    res.add(false);
                    continue search;
                }
                if (seen[(nums[j] - min) / d]) {
                    res.add(false);
                    continue search;
                }
                seen[(nums[j] - min) / d] = true;
            }
            res.add(true);
        }
        return res;

    }

    // 256. 粉刷房子 (Paint House) --plus 记忆化搜索
    // 剑指 Offer II 091. 粉刷房子
    private int[][] memo256;
    private int[][] costs256;

    public int minCost(int[][] costs) {
        int n = costs.length;
        this.costs256 = costs;
        memo256 = new int[n][3];
        for (int i = 0; i < n; ++i) {
            Arrays.fill(memo256[i], Integer.MAX_VALUE);
        }
        int res = Integer.MAX_VALUE;
        for (int type = 0; type < 3; ++type) {
            res = Math.min(res, dfs256(0, type));
        }
        return res;

    }

    private int dfs256(int i, int type) {
        if (i == costs256.length - 1) {
            return costs256[i][type];
        }
        if (memo256[i][type] != Integer.MAX_VALUE) {
            return memo256[i][type];
        }
        int min = Integer.MAX_VALUE;
        for (int j = 0; j < 3; ++j) {
            if (j != type) {
                min = Math.min(min, dfs256(i + 1, j));
            }
        }
        return memo256[i][type] = min + costs256[i][type];
    }

    // 256. 粉刷房子 (Paint House) --plus dp
    // 剑指 Offer II 091. 粉刷房子
    public int minCost2(int[][] costs) {
        int[][] dp = new int[costs.length][3];
        dp[0][0] = costs[0][0];
        dp[0][1] = costs[0][1];
        dp[0][2] = costs[0][2];
        for (int i = 1; i < costs.length; ++i) {
            dp[i][0] = Math.min(dp[i - 1][1], dp[i - 1][2]) + costs[i][0];
            dp[i][1] = Math.min(dp[i - 1][0], dp[i - 1][2]) + costs[i][1];
            dp[i][2] = Math.min(dp[i - 1][0], dp[i - 1][1]) + costs[i][2];
        }
        return Math.min(Math.min(dp[costs.length - 1][0], dp[costs.length - 1][1]), dp[costs.length - 1][2]);

    }

    // 256. 粉刷房子 (Paint House) --plus dp
    // 剑指 Offer II 091. 粉刷房子
    public int minCost3(int[][] costs) {
        int[] dp = costs[0];
        for (int i = 1; i < costs.length; ++i) {
            int[] newDp = new int[3];
            for (int j = 0; j < 3; ++j) {
                newDp[j] = Math.min(dp[(j + 1) % 3], dp[(j + 2) % 3]) + costs[i][j];
            }
            dp = newDp;
        }
        return Arrays.stream(dp).min().getAsInt();

    }

    // 1816. 截断句子 (Truncate Sentence)
    public String truncateSentence(String s, int k) {
        String[] strings = s.split("\\s+");
        StringBuilder res = new StringBuilder();
        for (int i = 0; i < k; ++i) {
            res.append(strings[i]).append(" ");
        }
        res.setLength(res.length() - 1);
        return res.toString();

    }

    // 1816. 截断句子 (Truncate Sentence)
    public String truncateSentence2(String s, int k) {
        int count = 0;
        char[] chars = s.toCharArray();
        for (int i = 0; i < chars.length; ++i) {
            if (chars[i] == ' ' && ++count == k) {
                return s.substring(0, i);
            }
        }
        return s;

    }

    // 1935. 可以输入的最大单词数 (Maximum Number of Words You Can Type)
    public int canBeTypedWords(String text, String brokenLetters) {
        int u = 0;
        for (char c : brokenLetters.toCharArray()) {
            u |= 1 << (c - 'a');
        }
        int res = 0;
        search: for (String s : text.split(" ")) {
            for (char c : s.toCharArray()) {
                if (((u >> (c - 'a')) & 1) != 0) {
                    continue search;
                }
            }
            ++res;
        }
        return res;

    }

    // 1347. 制造字母异位词的最小步骤数 (Minimum Number of Steps to Make Two Strings Anagram)
    public int minSteps(String s, String t) {
        int[] counts = new int[26];
        for (char c : t.toCharArray()) {
            ++counts[c - 'a'];
        }
        for (char c : s.toCharArray()) {
            --counts[c - 'a'];
        }
        int res = 0;
        for (int count : counts) {
            res += Math.abs(count);
        }
        return res / 2;

    }

    // 1207. 独一无二的出现次数 (Unique Number of Occurrences)
    public boolean uniqueOccurrences(int[] arr) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : arr) {
            map.put(num, map.getOrDefault(num, 0) + 1);
        }
        Set<Integer> set = new HashSet<>(map.values());
        return map.values().size() == set.size();

    }

    // 890. 查找和替换模式 (Find and Replace Pattern)
    public List<String> findAndReplacePattern(String[] words, String pattern) {
        List<String> res = new ArrayList<>();
        for (String word : words) {
            if (isSamePattern(word, pattern)) {
                res.add(word);
            }

        }
        return res;

    }

    private boolean isSamePattern(String word, String pattern) {
        Map<Character, Character> map1 = new HashMap<>();
        Map<Character, Character> map2 = new HashMap<>();
        char[] charWords = word.toCharArray();
        char[] patternWords = pattern.toCharArray();
        for (int i = 0; i < charWords.length; ++i) {
            if (!map1.containsKey(charWords[i])) {
                map1.put(charWords[i], patternWords[i]);
            } else if (map1.get(charWords[i]) != patternWords[i]) {
                return false;
            }

            if (!map2.containsKey(patternWords[i])) {
                map2.put(patternWords[i], charWords[i]);
            } else if (map2.get(patternWords[i]) != charWords[i]) {
                return false;
            }

        }

        return true;
    }

    // 893. 特殊等价字符串组 (Groups of Special-Equivalent Strings)
    public int numSpecialEquivGroups(String[] words) {
        Set<String> set = new HashSet<>();
        for (String word : words) {
            String even = "";
            String odd = "";
            for (int i = 0; i < word.length(); ++i) {
                if ((i & 1) == 1) {
                    odd += word.charAt(i);
                } else {
                    even += word.charAt(i);
                }
            }
            char[] evenChars = even.toCharArray();
            char[] oddChars = odd.toCharArray();
            Arrays.sort(evenChars);
            Arrays.sort(oddChars);
            set.add(String.valueOf(evenChars) + String.valueOf(oddChars));
        }
        return set.size();

    }

    // 1418. 点菜展示表 (Display Table of Food Orders in a Restaurant)
    public List<List<String>> displayTable(List<List<String>> orders) {
        Set<String> foodName = new HashSet<>();
        Map<Integer, Map<String, Integer>> tableName = new HashMap<>();
        for (List<String> order : orders) {
            String food = order.get(2);
            foodName.add(food);
            int tableId = Integer.parseInt(order.get(1));
            tableName.put(tableId, tableName.getOrDefault(tableId, new HashMap<>()));
            Map<String, Integer> map = tableName.get(tableId);
            map.put(food, map.getOrDefault(food, 0) + 1);
        }
        List<String> nameList = new ArrayList<>();
        for (String food : foodName) {
            nameList.add(food);
        }
        Collections.sort(nameList);
        List<Integer> tableList = new ArrayList<>();
        for (int table : tableName.keySet()) {
            tableList.add(table);
        }
        Collections.sort(tableList);
        List<List<String>> res = new ArrayList<>();
        List<String> firstRow = new ArrayList<>();
        firstRow.add("Table");
        firstRow.addAll(nameList);
        res.add(firstRow);
        for (int i = 0; i < tableList.size(); ++i) {
            List<String> row = new ArrayList<>();
            row.add(Integer.toString(tableList.get(i)));
            Map<String, Integer> map = tableName.get(tableList.get(i));
            for (int j = 0; j < nameList.size(); ++j) {
                row.add(String.valueOf(map.getOrDefault(nameList.get(j), 0)));
            }
            res.add(row);
        }
        return res;

    }

    // 884. 两句话中的不常见单词 (Uncommon Words from Two Sentences)
    public String[] uncommonFromSentences(String s1, String s2) {
        List<String> res = new ArrayList<>();
        Map<String, Integer> map = new HashMap<>();
        getUncommonWords(s1, map);
        getUncommonWords(s2, map);
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            if (entry.getValue() == 1) {
                res.add(entry.getKey());
            }
        }
        return res.toArray(new String[0]);

    }

    private void getUncommonWords(String s, Map<String, Integer> map) {
        String[] strs = s.split("\\s+");
        for (String str : strs) {
            map.put(str, map.getOrDefault(str, 0) + 1);
        }
    }

    // 1743. 从相邻元素对还原数组 (Restore the Array From Adjacent Pairs)
    public int[] restoreArray(int[][] adjacentPairs) {
        Map<Integer, List<Integer>> map = new HashMap<>();
        for (int[] adjacentPair : adjacentPairs) {
            map.computeIfAbsent(adjacentPair[0], k -> new ArrayList<>()).add(adjacentPair[1]);
            map.computeIfAbsent(adjacentPair[1], k -> new ArrayList<>()).add(adjacentPair[0]);
        }
        int[] res = new int[adjacentPairs.length + 1];
        for (Map.Entry<Integer, List<Integer>> entry : map.entrySet()) {
            int key = entry.getKey();
            if (entry.getValue().size() == 1) {
                res[0] = key;
                break;
            }
        }
        res[1] = map.get(res[0]).get(0);
        for (int i = 2; i < adjacentPairs.length + 1; ++i) {
            List<Integer> list = map.get(res[i - 1]);
            res[i] = list.get(0) == res[i - 2] ? list.get(1) : list.get(0);
        }
        return res;

    }

    // 791. 自定义字符串排序 (Custom Sort String)
    public String customSortString(String order, String s) {
        int[] counts = new int[26];
        for (char c : s.toCharArray()) {
            ++counts[c - 'a'];
        }
        StringBuilder res = new StringBuilder();
        for (char c : order.toCharArray()) {
            while (counts[c - 'a']-- > 0) {
                res.append(c);
            }
        }
        for (int i = 0; i < 26; ++i) {
            while (counts[i]-- > 0) {
                res.append((char) (i + 'a'));
            }
        }
        return res.toString();

    }

    // 1817. 查找用户活跃分钟数 (Finding the Users Active Minutes)
    public int[] findingUsersActiveMinutes(int[][] logs, int k) {
        // key : id
        // value : minutes
        Map<Integer, Set<Integer>> map = new HashMap<>();
        for (int[] log : logs) {
            map.computeIfAbsent(log[0], o -> new HashSet<>()).add(log[1]);
        }

        int[] res = new int[k];
        for (Set<Integer> set : map.values()) {
            ++res[set.size() - 1];
        }
        return res;

    }

    // 剑指 Offer II 066. 单词之和 --还需掌握前缀树的方法
    class MapSum {
        private Map<String, Integer> map;

        /** Initialize your data structure here. */
        public MapSum() {
            map = new HashMap<>();

        }

        public void insert(String key, int val) {
            map.put(key, val);
        }

        public int sum(String prefix) {
            int sum = 0;
            for (String key : map.keySet()) {
                if (isPrefix(key, prefix)) {
                    sum += map.get(key);
                }
            }
            return sum;

        }

        private boolean isPrefix(String key, String prefix) {
            if (prefix.length() > key.length()) {
                return false;
            }
            int i = 0;
            int j = 0;
            while (i < prefix.length()) {
                if (prefix.charAt(i) != key.charAt(j)) {
                    return false;
                }
                ++i;
                ++j;
            }
            return true;
        }
    }

    // 36. 有效的数独 (Valid Sudoku) --位运算
    public boolean isValidSudoku(char[][] board) {
        int n = board.length;
        int[] rows = new int[9];
        int[] cols = new int[9];
        int[] boxes = new int[9];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == '.') {
                    continue;
                }
                int d = board[i][j] - '1';
                if ((rows[i] >> d & 1) != 0) {
                    return false;
                }
                rows[i] |= 1 << d;
                if ((cols[j] >> d & 1) != 0) {
                    return false;
                }
                cols[j] |= 1 << d;
                int b = (i / 3) * 3 + j / 3;
                if ((boxes[b] >> d & 1) != 0) {
                    return false;
                }
                boxes[b] |= 1 << d;
            }
        }
        return true;
    }

    // 139. 单词拆分 (Word Break)
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        boolean[] dp = new boolean[s.length() + 1];
        dp[0] = true;
        for (int i = 1; i < dp.length; ++i) {
            for (int j = 0; j < i; ++j) {
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[dp.length - 1];

    }

    // 139. 单词拆分 (Word Break)
    private int n139;
    private String s139;
    private int[] memo139;
    private Trie139 trie139;

    public boolean wordBreak2(String s, List<String> wordDict) {
        this.n139 = s.length();
        this.s139 = s;
        this.memo139 = new int[n139];
        this.trie139 = new Trie139();
        for (String d : wordDict) {
            trie139.insert(d);
        }
        return dfs139(0);

    }

    private boolean dfs139(int i) {
        if (i == n139) {
            return true;
        }
        if (memo139[i] != 0) {
            return memo139[i] > 0;
        }
        for (int j : trie139.check(s139.substring(i))) {
            if (dfs139(i + j + 1)) {
                memo139[i] = 1;
                return true;
            }
        }
        memo139[i] = -1;
        return false;
    }

    class Trie139 {
        private boolean isEnd;
        private Trie139[] children;

        public Trie139() {
            this.children = new Trie139[26];
        }

        public void insert(String s) {
            Trie139 node = this;
            for (char c : s.toCharArray()) {
                int index = c - 'a';
                if (node.children[index] == null) {
                    node.children[index] = new Trie139();
                }
                node = node.children[index];
            }
            node.isEnd = true;
        }

        public List<Integer> check(String s) {
            Trie139 node = this;
            List<Integer> res = new ArrayList<>();
            for (int i = 0; i < s.length(); ++i) {
                int index = s.charAt(i) - 'a';
                if (node.children[index] == null) {
                    break;
                }
                node = node.children[index];
                if (node.isEnd) {
                    res.add(i);
                }
            }
            return res;
        }
    }

    // 819. 最常见的单词 (Most Common Word)
    public String mostCommonWord(String paragraph, String[] banned) {
        Set<String> set = new HashSet<>(Arrays.asList(banned));
        Map<String, Integer> map = new HashMap<>();
        int left = 0;
        int right = 0;
        char[] chars = paragraph.toCharArray();
        while (left < chars.length) {
            while (left < chars.length && !Character.isLetter(chars[left])) {
                ++left;
            }
            right = left;
            while (right < chars.length && Character.isLetter(chars[right])) {
                ++right;
            }
            if (left == right) {
                break;
            }
            String key = paragraph.substring(left, right).toLowerCase();
            if (!set.contains(key)) {
                map.put(key, map.getOrDefault(key, 0) + 1);
            }
            left = right;
        }
        int count = 0;
        String res = "";
        for (Map.Entry<String, Integer> entry : map.entrySet()) {
            if (entry.getValue() > count) {
                res = entry.getKey();
                count = entry.getValue();
            }
        }
        return res;

    }

    // 554. 砖墙 (Brick Wall)
    public int leastBricks(List<List<Integer>> wall) {
        Map<Integer, Integer> map = new HashMap<>();
        for (List<Integer> row : wall) {
            int sum = 0;
            for (int i = 0; i < row.size() - 1; ++i) {
                sum += row.get(i);
                map.put(sum, map.getOrDefault(sum, 0) + 1);
            }
        }
        if (map.size() == 0) {
            return wall.size();
        }
        return wall.size() - Collections.max(map.values());

    }

    // 535. TinyURL 的加密与解密 (Encode and Decode TinyURL)
    public class Codec {
        private Map<String, String> map = new HashMap<>();

        // Encodes a URL to a shortened URL.
        public String encode(String longUrl) {
            String key = "http://tinyurl.com/" + longUrl.hashCode();
            map.put(key, longUrl);
            return key;
        }

        // Decodes a shortened URL to its original URL.
        public String decode(String shortUrl) {
            return map.get(shortUrl);
        }
    }

    // 535. TinyURL 的加密与解密 (Encode and Decode TinyURL)
    public class Codec2 {
        private String code = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        private int count = 1;
        private Map<String, String> map = new HashMap<>();

        private String getString() {
            int c = count;
            StringBuilder builder = new StringBuilder();
            while (c-- > 0) {
                builder.append(code.charAt(c % 62));
                c /= 62;
            }
            return builder.toString();

        }

        // Encodes a URL to a shortened URL.
        public String encode(String longUrl) {
            String key = getString();
            map.put(key, longUrl);
            ++count;
            return "http://tinyurl.com/" + key;
        }

        // Decodes a shortened URL to its original URL.
        public String decode(String shortUrl) {
            return map.get(shortUrl.replace("http://tinyurl.com/", ""));
        }
    }

    // 645. 错误的集合 (Set Mismatch)
    public int[] findErrorNums(int[] nums) {
        int xor = 0;
        for (int i = 0; i < nums.length; ++i) {
            xor = xor ^ (i + 1) ^ nums[i];
        }
        int lowBit = xor & (-xor);
        int num1 = 0;
        int num2 = 0;
        for (int num : nums) {
            if ((num & lowBit) == 0) {
                num1 ^= num;
            } else {
                num2 ^= num;
            }
        }
        for (int i = 1; i <= nums.length; ++i) {
            if ((i & lowBit) == 0) {
                num1 ^= i;
            } else {
                num2 ^= i;
            }
        }
        for (int num : nums) {
            if (num == num1) {
                return new int[] { num1, num2 };
            }
        }
        return new int[] { num2, num1 };

    }

    // 2037. 使每位学生都有座位的最少移动次数 (Minimum Number of Moves to Seat Everyone)
    public int minMovesToSeat(int[] seats, int[] students) {
        Arrays.sort(seats);
        Arrays.sort(students);
        int res = 0;
        for (int i = 0; i < seats.length; ++i) {
            res += Math.abs(seats[i] - students[i]);
        }
        return res;

    }

    // 1791. 找出星型图的中心节点 (Find Center of Star Graph)
    public int findCenter(int[][] edges) {
        int p1 = edges[0][0];
        int p2 = edges[0][1];
        int p3 = edges[1][0];
        int p4 = edges[1][1];
        if (p1 == p3 || p1 == p4) {
            return p1;
        } else {
            return p2;
        }
    }

    // 740. 删除并获得点数 (Delete and Earn)
    private int mx740;
    private int[] cnt740;
    private int[] memo740;

    public int deleteAndEarn(int[] nums) {
        for (int x : nums) {
            mx740 = Math.max(mx740, x);
        }
        this.cnt740 = new int[mx740 + 1];
        for (int x : nums) {
            ++cnt740[x];
        }
        this.memo740 = new int[mx740 + 1];
        Arrays.fill(memo740, -1);
        return dfs740(1);

    }

    private int dfs740(int i) {
        if (i > mx740) {
            return 0;
        }
        if (memo740[i] != -1) {
            return memo740[i];
        }
        return memo740[i] = Math.max(dfs740(i + 1), dfs740(i + 2) + cnt740[i] * i);
    }

    // 748. 最短补全词 (Shortest Completing Word)
    public String shortestCompletingWord(String licensePlate, String[] words) {
        int[] counts = new int[26];
        for (char c : licensePlate.toCharArray()) {
            if (Character.isLetter(c)) {
                ++counts[Character.toLowerCase(c) - 'a'];
            }
        }
        String res = "";
        int[] curCounts = new int[26];
        for (String word : words) {
            for (char c : word.toCharArray()) {
                ++curCounts[c - 'a'];
            }
            boolean flag = false;
            for (int i = 0; i < counts.length; ++i) {
                if (curCounts[i] < counts[i]) {
                    flag = true;
                    break;
                }
            }
            if (!flag && (res.length() == 0 || word.length() < res.length())) {
                res = word;
            }
            Arrays.fill(curCounts, 0);
        }
        return res;

    }

    // 822. 翻转卡片游戏 (Card Flipping Game)
    public int flipgame(int[] fronts, int[] backs) {
        Set<Integer> set = new HashSet<>();
        for (int i = 0; i < fronts.length; ++i) {
            if (fronts[i] == backs[i]) {
                set.add(fronts[i]);
            }
        }
        int res = 2001;
        for (int front : fronts) {
            if (!set.contains(front)) {
                res = Math.min(res, front);
            }
        }
        for (int back : backs) {
            if (!set.contains(back)) {
                res = Math.min(res, back);
            }
        }
        return res == 2001 ? 0 : res;

    }

    // 997. 找到小镇的法官 (Find the Town Judge)
    public int findJudge(int n, int[][] trust) {
        int[] trusts = new int[n + 1];
        for (int[] t : trust) {
            --trusts[t[0]];
            ++trusts[t[1]];
        }
        for (int i = 1; i < trusts.length; ++i) {
            if (trusts[i] == n - 1) {
                return i;
            }
        }
        return -1;

    }

    // 300. 最长递增子序列 (Longest Increasing Subsequence)
    private int[] memo300;
    private int n300;
    private int[] nums300;

    public int lengthOfLIS(int[] nums) {
        this.n300 = nums.length;
        this.memo300 = new int[n300];
        this.nums300 = nums;
        int res = 0;
        for (int i = 0; i < n300; ++i) {
            res = Math.max(res, dfs300(i));
        }
        return res;

    }

    private int dfs300(int i) {
        if (memo300[i] != 0) {
            return memo300[i];
        }
        int max = 0;
        for (int j = 0; j < i; ++j) {
            if (nums300[j] < nums300[i]) {
                max = Math.max(max, dfs300(j));
            }
        }
        return memo300[i] = max + 1;
    }

    // 300. 最长递增子序列 (Longest Increasing Subsequence)
    public int lengthOfLIS2(int[] nums) {
        int res = 0;
        int n = nums.length;
        // dp表示以dp[i]结尾的最长严格递增序列的长度
        int[] dp = new int[n];
        for (int i = 0; i < n; ++i) {
            dp[i] = 1;
            for (int j = 0; j < i; ++j) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(dp[i], res);
        }
        return res;

    }

    // 300. 最长递增子序列 (Longest Increasing Subsequence) --二分
    public int lengthOfLIS3(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n + 1];
        int len = 1;
        dp[len] = nums[0];
        for (int i = 1; i < n; ++i) {
            if (nums[i] > dp[len]) {
                dp[++len] = nums[i];
            } else {
                int left = 1;
                int right = len;
                int pos = 0;
                while (left <= right) {
                    int mid = left + ((right - left) >>> 1);
                    if (dp[mid] < nums[i]) {
                        pos = mid;
                        left = mid + 1;
                    } else {
                        right = mid - 1;
                    }
                }
                dp[pos + 1] = nums[i];
            }

        }
        return len;

    }

    // 300. 最长递增子序列 (Longest Increasing Subsequence) --线段树
    private int[] seg300;

    public int lengthOfLIS4(int[] nums) {
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int x : nums) {
            min = Math.min(min, x);
            max = Math.max(max, x);
        }
        if (min < 2) {
            int add = 2 - min;
            for (int i = 0; i < nums.length; ++i) {
                nums[i] += add;
                max = Math.max(max, nums[i]);
            }
        }
        this.seg300 = new int[Integer.bitCount(max) == 1 ? max << 1 : Integer.highestOneBit(max) << 2];
        for (int x : nums) {
            int cur = query300(1, 1, max, 1, x - 1) + 1;
            modify300(1, 1, max, x, cur);
        }
        return seg300[1];

    }

    private void modify300(int o, int l, int r, int id, int val) {
        if (l == r) {
            seg300[o] = val;
            return;
        }
        int mid = l + ((r - l) >> 1);
        if (id <= mid) {
            modify300(o * 2, l, mid, id, val);
        } else {
            modify300(o * 2 + 1, mid + 1, r, id, val);
        }
        seg300[o] = Math.max(seg300[o * 2], seg300[o * 2 + 1]);
    }

    private int query300(int o, int l, int r, int L, int R) {
        if (L <= l && r <= R) {
            return seg300[o];
        }
        int mid = l + ((r - l) >> 1);
        int max = 0;
        if (L <= mid) {
            max = query300(o * 2, l, mid, L, R);
        }
        if (R >= mid + 1) {
            max = Math.max(max, query300(o * 2 + 1, mid + 1, r, L, R));
        }
        return max;
    }

    // 300. 最长递增子序列 (Longest Increasing Subsequence) --动态开点线段树
    public int lengthOfLIS5(int[] nums) {
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for (int x : nums) {
            min = Math.min(min, x);
            max = Math.max(max, x);
        }
        SegNode300 root = new SegNode300(min, max);
        for (int x : nums) {
            int cur = query300(root, min, x - 1) + 1;
            modify300(root, x, cur);
        }
        return root.val;
    }

    private int query300(SegNode300 node, int L, int R) {
        if (node == null) {
            return 0;
        }
        if (L > node.hi || R < node.lo) {
            return 0;
        }
        if (L <= node.lo && node.hi <= R) {
            return node.val;
        }
        return Math.max(query300(node.left, L, R), query300(node.right, L, R));
    }

    private void modify300(SegNode300 node, int x, int val) {
        node.val = Math.max(node.val, val);
        if (node.lo == node.hi) {
            return;
        }
        int mid = node.lo + ((node.hi - node.lo) >> 1);
        if (x <= mid) {
            if (node.left == null) {
                node.left = new SegNode300(node.lo, mid);
            }
            modify300(node.left, x, val);
        } else {
            if (node.right == null) {
                node.right = new SegNode300(mid + 1, node.hi);
            }
            modify300(node.right, x, val);
        }
    }

    public class SegNode300 {
        public int lo;
        public int hi;
        public int val;
        public SegNode300 left;
        public SegNode300 right;

        public SegNode300(int lo, int hi) {
            this.lo = lo;
            this.hi = hi;
        }

    }

    // 1357. 每隔 n 个顾客打折 (Apply Discount Every n Orders)
    class Cashier {
        private int n;
        private int discount;
        private int[] map;
        private int count;

        public Cashier(int n, int discount, int[] products, int[] prices) {
            this.n = n;
            this.discount = discount;
            this.map = new int[201];
            for (int i = 0; i < products.length; ++i) {
                map[products[i]] = prices[i];
            }
        }

        public double getBill(int[] product, int[] amount) {
            int sum = 0;
            for (int i = 0; i < product.length; ++i) {
                sum += map[product[i]] * amount[i];
            }
            if ((++count) % n == 0) {
                return (double) sum * (100 - discount) / 100;
            }
            return sum;
        }
    }

    // 1624. 两个相同字符之间的最长子字符串 (Largest Substring Between Two Equal Characters)
    public int maxLengthBetweenEqualCharacters(String s) {
        int res = -1;
        int[] pos = new int[26];
        Arrays.fill(pos, -1);
        for (int i = 0; i < s.length(); ++i) {
            int index = s.charAt(i) - 'a';
            if (pos[index] == -1) {
                pos[index] = i;
            } else {
                res = Math.max(res, i - pos[index] - 1);
            }
        }
        return res;
    }

    // 1309. 解码字母到整数映射 (Decrypt String from Alphabet to Integer Mapping)
    public String freqAlphabets(String s) {
        StringBuilder res = new StringBuilder();
        int i = s.length() - 1;
        while (i >= 0) {
            if (s.charAt(i) == '#') {
                res.append((char) (Integer.parseInt(s.substring(i - 2, i)) + 'a' - 1));
                i -= 3;
            } else {
                res.append((char) (Integer.parseInt(String.valueOf(s.charAt(i--))) + 'a' - 1));
            }
        }
        return res.reverse().toString();

    }

    // 剑指 Offer II 032. 有效的变位词
    // 242. 有效的字母异位词 (Valid Anagram)
    public boolean isAnagram(String s, String t) {
        if (s.equals(t)) {
            return false;
        }
        int[] sCounts = new int[26];
        for (char c : s.toCharArray()) {
            ++sCounts[c - 'a'];
        }
        int[] tCounts = new int[26];
        for (char c : t.toCharArray()) {
            ++tCounts[c - 'a'];
        }
        return Arrays.equals(sCounts, tCounts);
    }

    // 953. 验证外星语词典 (Verifying an Alien Dictionary)
    // 剑指 Offer II 034. 外星语言是否排序
    public boolean isAlienSorted(String[] words, String order) {
        int[] index = new int[26];
        for (int i = 0; i < order.length(); ++i) {
            index[order.charAt(i) - 'a'] = i;
        }
        search: for (int i = 0; i < words.length - 1; ++i) {
            String word1 = words[i];
            String word2 = words[i + 1];
            for (int j = 0; j < Math.min(word1.length(), word2.length()); ++j) {
                if (word1.charAt(j) != word2.charAt(j)) {
                    if (index[word1.charAt(j) - 'a'] > index[word2.charAt(j) - 'a']) {
                        return false;
                    }
                    continue search;
                }
            }
            if (word1.length() > word2.length()) {
                return false;
            }
        }
        return true;

    }

}
