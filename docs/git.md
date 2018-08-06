# 配置sshkey

取消全局变量设置
$ git config --global --unset user.name

$ git config --global --unset user.email

**配置用户**
$ git config user.name 'rockey2016'

$ git config user.email 'rockey-star@163.com'

# 下拉代码库

`$ git clone git@gitlab.dfc.com:MachineLearning/health_predict.git`
Cloning into 'health_predict'...

<!--克隆完成之后，可以看见一些git相关文件，实际上Git自动clone的是远程的master分支，并且把本地的master分支和远程的master分支对应起来。-->

`$ git branch -a`  <!--查看所有远程分支-->

* master
  remotes/origin/HEAD -> origin/master
  remotes/origin/develop
  remotes/origin/master

`$ git branch develop`  <!--创建远程origin的develop分支到本地--> 

$ `git branch`  <!--查看本地分支develop-->
  develop
* master

`$ git checkout develop`  <!--切换到本地develop分支-->
Switched to branch 'develop'

`$ git pull origin develop`  <!--从远程获取最新版本并merge到本地 ，git pull 相当于git fetch 和 git merge。在实际使用中，git fetch更安全一些，因为在merge前，我们可以查看更新情况，然后再决定是否合并 。-->



**拉取远程分支并创建本地分支，可以直接使用命令：**

`git checkout -b 本地分支名x origin/远程分支名x`

`$ git branch -r`
  origin/HEAD -> origin/master
  origin/gh-pages
  origin/master
  origin/testing_and_release


`$ git checkout -b gh-pages origin/gh-pages`
Switched to a new branch 'gh-pages'
Branch 'gh-pages' set up to track remote branch 'gh-pages' from 'origin'.

# 添加develop分支

`$ git checkout -b develop master`
Switched to a new branch 'develop'


`$ git branch`

* develop
  master

切换到Master分支 

`$ git checkout master`
Switched to branch 'master'
Your branch is up to date with 'origin/master'.

对Develop分支进行合并 ，将develop分支合并到master分支上。

`$ git merge --no-ff develop`
Already up to date.

<!--使用--no-ff参数后，会执行正常合并，在Master分支上生成一个新节点。为了保证版本演进的清晰，我们希望采用这种做法。--> 

`$ git push origin develop`  #将develop分支同步到远程服务器
Counting objects: 6, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (6/6), done.
Writing objects: 100% (6/6), 269.47 KiB | 8.69 MiB/s, done.
Total 6 (delta 3), reused 0 (delta 0)
remote:
remote: To create a merge request for develop, visit:
remote:   http://gitlab.dfc.com/MachineLearning/health_predict/merge_requests/new?merge_request%5Bsource_branch%5D=develop
remote:
To gitlab.dfc.com:MachineLearning/health_predict.git

 * [new branch]      develop -> develop

# 版本回退

###  git本地版本回退

`$ git log -oneline`
13b55aa (HEAD -> develop, origin/master, master) 20180717
80400c4 Merge branch 'master' of gitlab.dfc.com:MachineLearning/health_predict not correpondse
0263f7c commit at 20180717
d7905de 格式调整
4fe4fbc Add new file .gitignore
0706784 update at 20180706
adb2f19 20180629
ad538d6 commit update at 20180628
421a3a3 commit by sxx at 20180615
b83396c add readme file
81f0939 first commit at 20180614 by sxx

`$ git reset --hard d7905de`
HEAD is now at d7905de 格式调整

`$ git log --oneline`

d7905de (HEAD -> develop) 格式调整

4fe4fbc Add new file .gitignore
0706784 update at 20180706
adb2f19 20180629
ad538d6 commit update at 20180628
421a3a3 commit by sxx at 20180615
b83396c add readme file
81f0939 first commit at 20180614 by sxx



### git远程版本回退

`git push origin HEAD --force` #远程提交回退 

或者

`git reset --hard HEAD~1`

`git push --force` 



# 添加远程仓库

$ git remote add github git@github.com:rockey2016/Health.git



