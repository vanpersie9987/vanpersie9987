import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.stream.Collectors;

public class Leetcode_10 {

    // 3688. 偶数的按位或运算 (Bitwise OR of Even Numbers in an Array)
    public int evenNumberBitwiseORs(int[] nums) {
        int res = 0;
        for (int x : nums) {
            if ((x & 1) == 0) {
                res |= x;
            }
        }
        return res;

    }

    // 3689. 最大子数组总值 I (Maximum Total Subarray Value I)
    public long maxTotalValue(int[] nums, int k) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        for (int x : nums) {
            max = Math.max(max, x);
            min = Math.min(min, x);
        }
        return (long) (max - min) * k;

    }

    // 3690. 拆分合并数组 (Split and Merge Array Transformation)
    public int minSplitMerge(int[] nums1, int[] nums2) {
        int n = nums1.length;
        List<Integer> nums2List = toList3690(nums2);
        Set<List<Integer>> vis = new HashSet<>();
        vis.add(toList3690(nums1));
        Queue<List<Integer>> q = new LinkedList<>();
        q.add(toList3690(nums1));
        int res = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            for (int s = 0; s < size; ++s) {
                List<Integer> a = q.poll();
                if (a.equals(nums2List)) {
                    return res;
                }
                for (int l = 0; l < n; l++) {
                    for (int r = l + 1; r <= n; r++) {
                        List<Integer> sub = a.subList(l, r);
                        List<Integer> b = new ArrayList<>(a);
                        b.subList(l, r).clear(); // 从 b 中移除 sub
                        for (int i = 0; i <= b.size(); i++) {
                            List<Integer> c = new ArrayList<>(b);
                            c.addAll(i, sub);
                            if (vis.add(c)) { // c 不在 vis 中
                                q.add(c);
                            }
                        }
                    }
                }
            }
            ++res;
        }
        return -1;
    }

    private List<Integer> toList3690(int[] nums) {
        return Arrays.stream(nums).boxed().collect(Collectors.toList());
    }

    // 166. 分数到小数 (Fraction to Recurring Decimal)
    public String fractionToDecimal(int numerator, int denominator) {
        long a = numerator;
        long b = denominator;
        String sign = a * b < 0 ? "-" : "";
        a = Math.abs(a);
        b = Math.abs(b);
        long q = a / b;
        long r = a % b;
        if (r == 0) {
            return sign + q;
        }
        StringBuilder res = new StringBuilder(sign).append(q).append(".");
        Map<Long, Integer> map = new HashMap<>();
        map.put(r, res.length());
        while (r != 0) {
            r *= 10;
            q = r / b;
            r %= b;
            res.append(q);
            if (map.containsKey(r)) {
                int index = map.get(r);
                res.insert(index, "(");
                res.append(")");
                break;
            }
            map.put(r, res.length());
        }
        return res.toString();
    }

    // 976. 三角形的最大周长 (Largest Perimeter Triangle)
    public int largestPerimeter(int[] nums) {
        Arrays.sort(nums);
        for (int i = nums.length - 1; i >= 2; --i) {
            if (nums[i] < nums[i - 1] + nums[i - 2]) {
                return nums[i] + nums[i - 1] + nums[i - 2];
            }
        }
        return 0;
    }

    // 3692. 众数频率字符 (Majority Frequency Characters)
    public String majorityFrequencyGroup(String s) {
        int[] cnts = new int[26];
        int mxFreq = 0;
        for (char c : s.toCharArray()) {
            ++cnts[c - 'a'];
            mxFreq = Math.max(mxFreq, cnts[c - 'a']);
        }
        int[] freqCnts = new int[mxFreq + 1];
        int mxLen = 0;
        for (int i = 0; i < 26; ++i) {
            if (cnts[i] > 0) {
                freqCnts[cnts[i]] |= (1 << i);
                mxLen = Math.max(mxLen, Integer.bitCount(freqCnts[cnts[i]]));
            }
        }
        int mask = 0;
        for (int f = mxFreq; f >= 1; --f) {
            if (Integer.bitCount(freqCnts[f]) == mxLen) {
                mask = freqCnts[f];
                break;
            }
        }
        StringBuilder sb = new StringBuilder();
        while (mask != 0) {
            int lb = Integer.numberOfTrailingZeros(mask);
            sb.append((char) (lb + 'a'));
            mask &= (mask - 1);
        }
        return sb.toString();
    }

    // 3693. 爬楼梯 II (Climbing Stairs II)
    private int[] costs3693;
    private int n3693;
    private int[] memo3693;

    public int climbStairs(int n, int[] costs) {
        ++n;
        this.costs3693 = new int[n];
        for(int i = 0; i < n - 1; ++i) {
            this.costs3693[i + 1] = costs[i];
        }
        this.n3693 = n;
        this.memo3693 = new int[n];
        return dfs3693(0);
    }

    private int dfs3693(int i) {
        if (i == n3693 - 1) {
            return 0;
        }
        if (memo3693[i] != 0) {
            return memo3693[i];
        }
        int res = Integer.MAX_VALUE;
        for (int j = i + 1; j < Math.min(n3693, i + 4); ++j) {
            res = Math.min(res, dfs3693(j) + costs3693[j] + (j - i) * (j - i));
        }
        return memo3693[i] = res;
    }
}
