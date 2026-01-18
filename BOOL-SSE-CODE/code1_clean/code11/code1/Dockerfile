# 使用官方的 Python 3.9 精简版镜像作为基础
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装 pypbc 及其依赖
# 首先安装 libpbc-dev 和 libgmp-dev 等系统库
RUN apt-get update && apt-get install -y \
  libpbc-dev \
  libgmp-dev \
  && rm -rf /var/lib/apt/lists/*

# 将所有项目文件复制到容器中
COPY . .

# 安装所有 Python 依赖
# 注意：这里我们将所有依赖都列出来，确保环境完整
RUN pip install --no-cache-dir pypbc numpy matplotlib pandas

# 设置容器启动时要运行的命令
# 我将入口文件设置为 demo.py，如果你的入口文件是 FrontTest.py，请自行修改
CMD ["python", "demo.py"]