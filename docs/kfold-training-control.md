# Label Studio K 折训练入口

该入口把 K 折训练控制页挂载到 Label Studio 同一地址下：

```text
Label Studio 项目页 -> K折训练 -> /kfold-training/
```

## 上传格式

在 Label Studio 的 Export 页面选择普通 `JSON`。不要选择 COCO、`JSON_MIN`
或包含模型 predictions 的其他格式。

上传转换器执行以下强制规则：

- 每个任务必须存在已提交且未取消的 annotation；未提交任务不会被当成空图；
- 只保留 `type=rectanglelabels`、`from_name=label` 且非只读的人工框；
- 删除 `audit_hint`、`prediction_reference`、`review_issue` 等非人工结果；
- 删除 `candidate_summary`、`review_tag` 和 `kfold_*` 等审计字段；
- 只接受八个训练类别；“可能漏标”等提示不能成为训练类别；
- 按完整源图片路径继承上一轮 `validation_fold`，不同相机不会因文件同名而混淆；
- 明确提交的零框任务会记录为空图，随后由 K 折准备阶段按既定规则排除。

转换报告记录任务数、人工框数、删除的提示数、空图数以及每折图片数。

## 训练约束

网页用户不能设置命令、镜像或文件路径。后台固定：

- Docker 镜像 `car5-rfdetr:v4.4-70px-train-user`；
- batch size 32；
- gradient accumulation 1；
- 最多 80 epochs，early stopping patience 15；
- 每次使用新的 workspace 和 artifact 目录；
- 不使用上一轮任何折的 checkpoint 作为共享预训练权重；
- 同时最多运行一个五折任务。

网页服务只写入受限任务队列，不挂载 Docker socket。一个不监听网络的本机 worker
以部署账号读取队列并启动固定训练命令。浏览器用户不需要 Linux、sudo 或 Docker
权限。训练页使用独立 HTTP Basic 凭据；密码只能通过环境
变量 `CAR5_KFOLD_TRAINING_PASSWORD` 注入，不能写入 Git 或命令行。

## 服务启动

控制服务：

```bash
CAR5_KFOLD_TRAINING_PASSWORD='<secret>' PYTHONPATH=src \
python3 tools/serve_kfold_training_control.py \
  --assignment-template /path/to/previous/assignments.csv \
  --image-root /path/to/dataset/root \
  --workspace-root /path/to/new/workspaces \
  --artifact-root /path/to/new/artifacts \
  --repository-root /path/to/2d-auto-labeling \
  --gpu 3 --gpu 4
```

控制服务监听 `8091`。`deploy/kfold-training-control/nginx.conf` 在同机 `8090`
对外入口中把 `/kfold-training/` 转发到控制服务，其余请求转发到 Label Studio，
并在项目页注入“K折训练”入口。

本机 worker：

```bash
PYTHONPATH=src python3 tools/run_kfold_training_worker.py \
  --jobs-root /path/to/artifacts/control_jobs \
  --image-root /path/to/dataset/root \
  --workspace-root /path/to/new/workspaces \
  --artifact-root /path/to/new/artifacts \
  --repository-root /path/to/2d-auto-labeling \
  --gpu 3 --gpu 4
```

worker 不监听任何端口，只接受网页转换器生成且通过校验的队列目录。同时只能运行
一个 worker 和一个五折任务。

当前部署使用 HTTP 内网访问，Basic 凭据在传输层不加密。若访问范围超出可信内网，
必须先为 8090 配置 HTTPS。回滚时停止训练控制容器并恢复原有纯 TCP 代理即可；
Label Studio 数据库和项目配置不需要修改。
