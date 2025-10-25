## venv
    ```bash
        curl -fsSL https://pyenv.run | bash
        export PATH="$HOME/.pyenv/bin:$PATH"
        pyenv install 3.12.7
        pyenv local 3.12.7
        python -m venv .venv
        source .venv/bin/activate
        pip install -r requirements.txt
    ```


## 背景執行 
- 每個 experiment 都用不同的 log/run_model**.log 和 log/run**.pid 檔案來區分
- 單 GPU
    `nohup python train.py > log/run_model1.log 2>&1 & echo $! > log/run1.pid &`
- 多 GPU (以 0,1 兩張卡為例)
    `nohup python -m torch.distributed.run --nproc_per_node=2 train.py > log/run_exp1_baseline_ddp.log 2>&1 & echo $! > log/run_exp1_ddp.pid &`
    `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --standalone train.py > log/run_model2.log 2>&1 & echo $! > log/run2.pid &`

## 檢查還在不在跑：
    way1. `ps -p "$(cat log/run1.pid)" -o pid,etime,cmd`      // 用 PID 查
    way2. `pgrep -a -f 'python .*train.py'`              // 或比對檔名

## 停止： 
- kill PID & remove related PID file
    `kill "$(cat log/run1.pid)" && rm log/run1.pid`

## 注意
1. 先處理 ==git repo== 
2. 除了做 K-fold 的 validation set，記得要在 training set 切出 ==pseudo-test data== (5~10%)
3. 每次改 code 後，記得要改 train.py 裡的 CONFIG['note']，以利追蹤

# TODO
- [ ] model / dataloader / evaluate
- [ ] adaptive Learning Rate (LR)
- [ ] K-fold
- [ ] Checkpoint 存取
- [ ] JSON/CSV 存取
- [ ] 基本的 logging
- [ ] seed / device / DDP / env
- [ ] Early Stopping
