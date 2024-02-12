// 2223. 构造字符串的总得分和 (Sum of Scores of Built Strings)
func sumScores(s string) int64 {
    n := len(s)
	z := make([]int, n)
	ans := n
	for i, l, r := 1, 0, 0; i < n; i++ {
        if i <= r {
            z[i] = min(z[i-l], r-i+1)
        }
		for i + z[i] < n && s[z[i]] == s[i + z[i]] {
			l, r = i, i + z[i]
			z[i]++
		}
		ans += z[i]
	}
	return int64(ans)

}

func min(a, b int) int { if a > b { return b }; return a }
func max(a, b int) int { if a < b { return b }; return a }