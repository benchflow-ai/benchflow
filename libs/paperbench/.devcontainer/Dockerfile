# 使用官方 Python 3.11 slim 版本作为基础镜像
FROM python:3.11-slim

# 安装常用依赖，如vim、git等（根据需要添加）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 创建一个非 root 用户，用于在容器内开发
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# 设置工作目录
WORKDIR /workspace

# 将用户切换为非 root 用户
USER $USERNAME

# 暴露容器内你可能使用的端口（如需）
EXPOSE 8000
