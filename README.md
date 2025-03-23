# leetcode

<!-- 问题：Failed to connect to github.com port 443 after 75029 ms: Couldn't connect to server -->
<!-- 解决方法：
// 查询自己有没有设置代理，没有按下回车什么内容都不会输出
git config --global http.proxy
// 设置代理的命令(这里要注意一下我开的代理，也就是我们用的vpn,占用的端口是7890，所以我写的7890，看下自己用的哪个端口)
git config --global http.proxy <http://127.0.0.1:55883>
// 不用的时候最好取消了，不知道为什么突然正常push不上去了，时好时坏的。取消代理的命令
git config --global --unset http.proxy 
-->