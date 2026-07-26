#!/bin/bash
# ==========================================
# 脚本：start.sh
# 功能：在独立的 tmux 窗口中启动 4进程 或 5进程 训练集群
# ==========================================

# 公共环境变量
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 定义虚拟环境激活路径
VENV_ACTIVATE="source .venv/bin/activate"

echo "请选择启动模式:"
echo "[1] 4进程集群 (Server + Middle1-2 + Client)"
echo "[2] 5进程集群 (Server + Middle1-3 + Client)"
read -p "请输入数字 (1 或 2): " choice

# 定义一个统一的 tmux 会话名称
SESSION_NAME="train_cluster"

# 如果已经存在同名会话，先将其关闭，防止冲突
tmux kill-session -t $SESSION_NAME 2>/dev/null

# 创建一个新的后台 tmux 会话（不自动附加）
tmux new-session -d -s $SESSION_NAME

case $choice in
    1)
        echo "[启动中] 4进程集群模式..."
        
        # 窗口 0: 启动 Server (Rank 0)
        tmux send-keys -t $SESSION_NAME:0 "$VENV_ACTIVATE && export RANK=0 LOCAL_RANK=0 WORLD_SIZE=4 && python scripts/train_pretrain_france_server1.py" C-m
        
        # 窗口 1: 启动 Middle 1 (Rank 1)
        sleep 10
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:1 "$VENV_ACTIVATE && export RANK=1 LOCAL_RANK=1 WORLD_SIZE=4 && python scripts/train_pretrain_france_middle_server1.py" C-m
        
        # 窗口 2: 启动 Middle 2 (Rank 2)
        sleep 20
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:2 "$VENV_ACTIVATE && export RANK=2 LOCAL_RANK=2 WORLD_SIZE=4 && python scripts/train_pretrain_france_middle_server2.py" C-m
        
        # 窗口 3: 启动 Client (Rank 3)
        sleep 30
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:3 "$VENV_ACTIVATE && export RANK=3 LOCAL_RANK=3 WORLD_SIZE=4 && python scripts/train_pretrain_france_client1.py" C-m
        ;;
        
    2)
        echo "[启动中] 5进程集群模式..."
        
        # 窗口 0: 启动 Server (Rank 0)
        tmux send-keys -t $SESSION_NAME:0 "$VENV_ACTIVATE && export RANK=0 LOCAL_RANK=0 WORLD_SIZE=5 && python scripts/train_pretrain_france_server.py" C-m
        
        # 窗口 1: 启动 Middle 1 (Rank 1)
        sleep 2
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:1 "$VENV_ACTIVATE && export RANK=1 LOCAL_RANK=1 WORLD_SIZE=5 && python scripts/train_pretrain_france_middle_server1.py" C-m
        
        # 窗口 2: 启动 Middle 2 (Rank 2)
        sleep 2
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:2 "$VENV_ACTIVATE && export RANK=2 LOCAL_RANK=2 WORLD_SIZE=5 && python scripts/train_pretrain_france_middle_server2.py" C-m
        
        # 窗口 3: 启动 Middle 3 (Rank 3)
        sleep 2
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:3 "$VENV_ACTIVATE && export RANK=3 LOCAL_RANK=3 WORLD_SIZE=5 && python scripts/train_pretrain_france_middle_server3.py" C-m
 
        # 窗口 4: 启动 Client 4 (Rank 4)
        sleep 2
        tmux new-window -t $SESSION_NAME
        tmux send-keys -t $SESSION_NAME:4 "$VENV_ACTIVATE && export RANK=4 LOCAL_RANK=4 WORLD_SIZE=5 && python scripts/train_pretrain_france_client.py" C-m
        ;;
        
    *)
        echo "无效的选择，退出脚本。"
        exit 1
        ;;
esac

echo "[完成] 所有进程已在独立的 tmux 窗口中启动！"
echo "提示：你可以使用命令 'tmux attach -t $SESSION_NAME' 进入后台查看，按 Ctrl+B 然后按 D 退出查看。"