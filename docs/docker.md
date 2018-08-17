使用官方脚本安装：

`curl -fsSL https://get.docker.com | bash -s docker --mirror Aliyun`

创建docker用户：

sudo usermod -aG docker docker

安装校验：

![1534422209971](D:\Manshy\OpenProject\MachineLearning\docs\assets\1534422209971.png)

配置加速器：

sudo mkdir -p /etc/docker 

sudo tee /etc/docker/daemon.json <<-'EOF' 

{ "registry-mirrors": ["https://2pltq3lu.mirror.aliyuncs.com"] } 

EOF 

sudo systemctl daemon-reload 

sudo systemctl restart docker 