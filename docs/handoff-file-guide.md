# V4.4 交接包文件说明

本说明对应 `car5-rfdetr-v44-70px` 交接包。标记含义：

- **核心**：复现 V4.4 推理必须重点检查的文件；
- **运行**：部署或运行所需；
- **训练**：继续训练时使用；
- **评测**：计算指标或检查结果时使用；
- **旧版**：早期 RTMDet/V1 方案，为项目历史和兼容性保留；
- **开发**：测试、协作和代码质量文件，不参与模型推理。

## 一、打开压缩包后先看这 12 项

| 文件 | 标记 | 作用 |
| --- | --- | --- |
| `weights/checkpoint_best_total.pth` | **核心** | V4.4 训练得到的模型权重；没有它就不能复现当前模型。 |
| `MODEL_RELEASE.json` | **核心** | 本交接包版本、Git commit、权重 SHA-256、70px规则和生成时间。 |
| `tools/infer_rfdetr_image_folder.py` | **核心** | V4.4 图片文件夹推理总入口。加载权重并串联小目标复核、去重、类别策略和70px过滤。 |
| `Dockerfile.rfdetr` | **核心** | 构建 Linux/NVIDIA GPU 推理容器。权重和图片在运行时挂载。 |
| `configs/models/rfdetr/mining8_v44_70px.json` | **核心** | 模型身份、7类顺序、权重文件名及哈希、参考环境和70px正式规则。 |
| `configs/policies/mining8_v44_class_thresholds_v1.json` | **核心** | 7个类别各自采用的置信度阈值。 |
| `configs/policies/annotation_v2_70px.json` | **核心** | 人工真值和正式预测统一采用的70px标注边界。 |
| `src/car5_autolabel/postprocessing.py` | **核心** | 同类重复框、跨类别重叠框和矿场类别规则的后处理。 |
| `src/car5_autolabel/roi_verification.py` | **核心** | 对低置信度小目标回到原图局部区域再次确认。 |
| `src/car5_autolabel/tiling.py` | **核心** | 高分辨率切片推理及切片框映射回原图。 |
| `src/car5_autolabel/policies/size.py` | **核心** | 按原图像素计算最长边并执行70px过滤。 |
| `docs/rfdetr-v44-70px.md` | **运行** | V4.4推理、评测、权重校验和交接命令。 |

## 二、根目录文件

| 文件 | 标记 | 作用 |
| --- | --- | --- |
| `.dockerignore` | 运行 | 防止构建镜像时把权重、图片、标注、数据库和输出复制进镜像。 |
| `.editorconfig` | 开发 | 统一编辑器的缩进、编码和换行格式。 |
| `.env.example` | 开发 | 环境变量示例，不含真实密码和Token。 |
| `.gitattributes` | 开发 | Git换行符及文件属性设置。 |
| `.gitignore` | 开发 | 禁止数据、权重、数据库和生成目录进入Git。 |
| `AGENTS.md` | 开发 | AI及开发者在本仓库内工作的约定。 |
| `CONTRIBUTING.md` | 开发 | 提交代码和协作流程说明。 |
| `README.md` | 运行 | 仓库首页、核心文件导航和快速开始。 |
| `SECURITY.md` | 开发 | 数据、凭据和模型资产的安全边界。 |
| `pyproject.toml` | 运行 | Python项目元数据、基础依赖、测试和Ruff配置。 |
| `requirements-rfdetr.txt` | 运行 | RF-DETR 1.8.3推理附加依赖。 |
| `requirements-gpu-v1.txt` | 旧版 | 早期RTMDet/MMDetection GPU依赖，不是V4.4 Docker的主依赖。 |
| `.github/pull_request_template.md` | 开发 | GitHub合并申请模板。 |

## 三、配置文件 `configs/`

| 文件 | 标记 | 作用 |
| --- | --- | --- |
| `configs/classes.yaml` | 运行 | 项目通用类别名称、ID和颜色定义。 |
| `configs/cameras/camera_policy_v1.yaml` | 运行 | 六路相机名称、顺序和相机策略。 |
| `configs/policies/taxonomy_v1.yaml` | 运行 | 类别体系和类别语义规则。 |
| `configs/policies/annotation_v1.yaml` | 旧版 | 70px规则确定前的V1标注规范。 |
| `configs/policies/annotation_v2_70px.json` | **核心** | 当前70px人工真值和评测规则。 |
| `configs/policies/mining8_v44_class_thresholds_v1.json` | **核心** | 当前V4.4分类别阈值。 |
| `configs/models/rfdetr/mining8_v44_70px.json` | **核心** | 当前V4.4模型发布配置。 |
| `configs/models/rtmdet/rtmdet_s_mining6_conservative_80e.py` | 旧版 | RTMDet-S保守策略80轮训练配置。 |
| `configs/models/rtmdet/rtmdet_s_mining6_conservative_smoke.py` | 旧版 | 上述RTMDet-S配置的快速冒烟测试版。 |
| `configs/models/rtmdet/rtmdet_s_mining6_reference_40e.py` | 旧版 | RTMDet-S参考40轮配置。 |
| `configs/models/rtmdet/rtmdet_tiny_mining6_40e.py` | 旧版 | RTMDet-Tiny 40轮训练配置。 |
| `configs/models/rtmdet/rtmdet_tiny_mining6_deploy_all_12e.py` | 旧版 | RTMDet-Tiny全量部署微调配置。 |
| `configs/models/rtmdet/rtmdet_tiny_mining6_fold0_conservative_ft_12e.py` | 旧版 | RTMDet-Tiny第0折保守微调配置。 |
| `configs/models/rtmdet/rtmdet_tiny_mining6_smoke.py` | 旧版 | RTMDet-Tiny快速冒烟测试配置。 |

## 四、核心 Python 包 `src/car5_autolabel/`

| 文件 | 标记 | 作用 |
| --- | --- | --- |
| `__init__.py` | 运行 | 声明主Python包。 |
| `config.py` | 运行 | 读取服务和路径环境配置。 |
| `schemas.py` | 运行 | 检测框等公共数据结构。 |
| `postprocessing.py` | **核心** | 重复框抑制、跨类冲突和类别后处理。 |
| `roi_verification.py` | **核心** | 小目标原图ROI二次确认。 |
| `tiling.py` | **核心** | 切片生成、边缘判断和坐标回映射。 |
| `api/__init__.py` | 开发 | 声明API子包。 |
| `api/main.py` | 运行 | FastAPI健康检查和服务入口；文件夹CLI推理不依赖启动该服务。 |
| `datasets/__init__.py` | 开发 | 声明数据集工具子包。 |
| `datasets/label_studio_export.py` | 运行 | 把预测转换为Label Studio可导入格式。 |
| `datasets/manifest.py` | 运行 | 构建图片清单和稳定样本ID。 |
| `datasets/models.py` | 运行 | 数据集记录的数据模型。 |
| `datasets/release.py` | 运行 | 生成数据发布清单和版本信息。 |
| `detectors/__init__.py` | 开发 | 声明检测器子包。 |
| `detectors/base.py` | 开发 | 检测器统一接口。 |
| `detectors/rtmdet.py` | 旧版 | RTMDet检测器适配器，不是RF-DETR V4.4主入口。 |
| `evaluation/__init__.py` | 开发 | 声明评测子包。 |
| `evaluation/detection.py` | 评测 | TP、FP、FN、Precision、Recall、F1和IoU匹配。 |
| `integrations/__init__.py` | 开发 | 声明外部系统集成子包。 |
| `integrations/label_studio.py` | 运行 | Label Studio任务及预测数据适配。 |
| `policies/__init__.py` | 开发 | 声明策略子包。 |
| `policies/size.py` | **核心** | 原图最长边尺寸计算与正式过滤。 |
| `tracking/__init__.py` | 开发 | 追踪模块预留接口；当前V4.4单图推理没有启用正式追踪器。 |

## 五、命令行工具 `tools/`

| 文件 | 标记 | 作用 |
| --- | --- | --- |
| `infer_rfdetr_image_folder.py` | **核心** | 当前RF-DETR V4.4批量推理入口。 |
| `train_rfdetr.py` | 训练 | RF-DETR跨平台训练入口。 |
| `apply_class_threshold_policy.py` | 运行 | 对原始预测应用分类别阈值。 |
| `evaluate_rfdetr_roi_verification.py` | 评测 | 比较ROI小目标复核前后的指标。 |
| `evaluate_size_policy_grid.py` | 评测 | 对50/60/70px等尺寸规则进行公平评测。 |
| `evaluate_predictions.py` | 评测 | 通用预测与真值评测入口。 |
| `export_label_studio_predictions.py` | 运行 | 将预测结果导出成Label Studio预标注任务。 |
| `render_prediction_samples.py` | 评测 | 将预测框画到样例图片上用于人工检查。 |
| `select_keyframe_benchmark.py` | 评测 | 按相机和时间选择去重关键帧。 |
| `build_manifest.py` | 运行 | 扫描图片并生成数据清单。 |
| `build_release.py` | 运行 | 生成可审计的数据发布记录。 |
| `package_rfdetr_handoff.py` | 开发 | 校验权重哈希并构建当前交接ZIP。 |
| `infer_rtmdet.py` | 旧版 | 旧RTMDet批量推理入口。 |

## 六、Docker和Windows启动脚本

| 文件 | 标记 | 作用 |
| --- | --- | --- |
| `Dockerfile.rfdetr` | **核心** | 当前RF-DETR V4.4 GPU容器定义。 |
| `run_v1_autolabel.bat` / `.ps1` | 旧版 | Windows下运行V1自动标注。 |
| `run_v1_1_autolabel.bat` / `.ps1` | 旧版 | Windows下运行V1.1自动标注。 |
| `train_v1_1_deploy.bat` / `.ps1` | 旧版 | Windows下训练V1.1部署模型。 |
| `train_v1_1_fold0_conservative.ps1` | 旧版 | Windows下执行V1.1第0折保守训练。 |

## 七、测试 `tests/`

这些文件不参与实际推理，但用于证明代码行为没有被修改坏。

| 文件 | 作用 |
| --- | --- |
| `tests/test_health.py` | 测试API健康检查。 |
| `tests/test_label_studio.py` | 测试Label Studio转换和接口。 |
| `tests/test_postprocessing.py` | 测试重复框和跨类别冲突处理。 |
| `tests/test_roi_verification.py` | 测试小目标ROI复核。 |
| `tests/test_tiling.py` | 测试切片及坐标回映射。 |
| `tests/test_annotation_policy_fixtures.py` | 测试标注规范样例。 |
| `tests/policies/test_size.py` | 测试70px尺寸边界。 |
| `tests/evaluation/test_detection.py` | 测试检测指标和IoU匹配。 |
| `tests/evaluation/test_sequence_filter.py` | 测试序列筛选规则。 |
| `tests/datasets/test_label_studio_export.py` | 测试Label Studio导出。 |
| `tests/datasets/test_manifest.py` | 测试数据清单生成。 |
| `tests/datasets/test_release.py` | 测试发布清单和哈希。 |
| `tests/detectors/test_rtmdet.py` | 测试旧RTMDet适配器。 |
| `tests/fixtures/annotations/README.md` | 解释合成测试标注。 |
| `tests/fixtures/annotations/disagreements.synthetic.jsonl` | 不含公司数据的人工分歧合成样例。 |
| `tests/fixtures/annotations/gold_manifest.synthetic.jsonl` | 不含公司数据的金标准清单合成样例。 |

各目录中的 `__init__.py` 仅用于声明Python测试包。

## 八、项目文档 `docs/`

| 文件 | 标记 | 作用 |
| --- | --- | --- |
| `handoff-file-guide.md` | 运行 | 当前这份交接包逐项说明。 |
| `rfdetr-v44-70px.md` | **核心** | V4.4命令和复现说明。 |
| `docker-deployment.md` | 运行 | Docker构建、GPU挂载和运行说明。 |
| `annotation-policy.md` | 文档 | 通用人工标注规范。 |
| `evaluation-policy.md` | 评测 | 指标、数据划分和验收原则。 |
| `label-studio-contract.md` | 运行 | Label Studio字段及导入导出契约。 |
| `architecture.md` | 文档 | 项目模块架构。 |
| `technical-route.md` | 文档 | 项目整体技术路线。 |
| `roadmap.md` | 文档 | 开发阶段和里程碑。 |
| `work-packages.md` | 文档 | WP任务划分。 |
| `collaboration.md` | 开发 | 两名开发者分支、评审和合并流程。 |
| `license-register.md` | 文档 | 第三方依赖许可证记录。 |
| `v1-user-guide.md` | 旧版 | V1使用说明。 |
| `v1.1-user-guide.md` | 旧版 | V1.1使用说明。 |
| `v1.1-evaluation.md` | 旧版 | V1.1评测记录。 |
| `hardware/rtx4060-laptop-profile.md` | 文档 | RTX 4060笔记本训练/推理资源说明。 |
| `audits/2026-07-16-car5-v1-data-audit.md` | 文档 | 早期数据审计记录。 |
| `experiments/2026-07-17-rtmdet-tiny-fold0.md` | 旧版 | RTMDet-Tiny第0折实验记录。 |
| `research/2026-07-2d-autolabeling-landscape.md` | 文档 | 2D自动标注模型和工具调研。 |
| `decisions/0001-v1-data-scope.md` | 文档 | 数据范围决策。 |
| `decisions/0002-v1-taxonomy.md` | 文档 | 类别体系决策。 |
| `decisions/0003-camera-order.md` | 文档 | 六路相机顺序决策。 |
| `decisions/0004-v1-split-and-release-policy.md` | 文档 | 数据划分与发布决策。 |
| `decisions/0005-formal-70px-box-policy.md` | **核心** | 70px正式规则的决策依据。 |
| `reviews/README.md` | 文档 | 技术评审目录说明。 |
| `reviews/technical-route-review-colleague.md` | 文档 | 同事对技术路线的独立评审。 |
| `reviews/technical-route-review-resolution.md` | 文档 | 技术路线评审结论及处理。 |
| `agent-prompts/colleague-route-review.md` | 开发 | 同事智能体技术路线评审任务。 |
| `agent-prompts/colleague-wp02-annotation-policy.md` | 开发 | 同事智能体WP-02标注策略任务。 |

## 九、到底需要下载什么

- **只在GitHub审查代码**：不用单独下载，点击README或PR中的核心文件链接。
- **本地检查全部代码**：克隆或下载整个 `dev/colleague` 分支。
- **直接运行V4.4**：需要全部代码、`checkpoint_best_total.pth`、Python或Docker环境，以及对方自己的图片目录。
- **不需要**：原训练图片、原测试集或本机Label Studio数据库；它们不是加载现有权重进行推理的必要条件。
