# 源码修改后回放旧检查点

2026-09-26：新增 `scripts/eval_from_source_archive.py`。新准备动作修改包内源码后，旧训练仍必须使用保存时的源码；没有关闭或放宽任何 source SHA256、依赖版本、scene/task、配置与 checkpoint 一致性校验。

## 用法

在本独立项目目录执行：

```bash
training_mjlab/.venv/bin/python scripts/eval_from_source_archive.py \
  --source-archive training_mjlab/source_versions/before_preparation_20260926.tar.gz \
  --checkpoint training_mjlab/outputs/sac-twinkle-shadow-sync-5m-20260925-171119-593333/final.pt \
  --mode evaluate \
  --output preview/outputs/diagnostics/my_old_final_evaluation.json
```

`--mode` 支持原入口的 `evaluate`、`preview`、`render`。`--config` 默认取检查点旁的 `config.json`；其余参数如 `--output`、`--episodes`、`--device`、渲染相机／音频参数原样转交归档内入口。相对输出路径仍相对于当前独立项目；不接受绕过源码重建的 `--scene`／`--task`。

渲染例子，将上面的 `--mode evaluate` 改为 `--mode render`，`--output` 改为一个尚不存在的输出目录。归档必须包含当时的 `preview/scripts/render_mjlab_checkpoint.py`；缺失时明确报错，绝不补用当前源码的预览脚本。

## 实现与边界

1. 只使用用户信任的本地源码归档：归档中的 Python 将被执行，路径过滤不等于对任意恶意源码的沙箱。
2. 验证 tar 所有成员后才解压。拒绝绝对路径、`..`、软／硬链接、设备等特殊文件及 `.venv` 成员；另使用 Python `tarfile` 的 `data` filter。
3. 解压到临时目录，包内全部源码／资源的文件集合及 SHA256 必须与 checkpoint 的 `source_runtime.source_sha256` 完全相同；多文件、少文件或内容变化都拒绝。
4. 临时目录通过软链接引用当前已安装的 `training/.venv`、`training_mjlab/.venv`；脚本不安装或修改环境。已有 `training_mjlab/audio_runtime` 也按需引用，提供 FluidSynth／SoundFont；若归档自带音频运行库则保留归档版本。音频库／音色属于包外运行依赖，实际 SoundFont 路径与 SHA256 仍写入视频报告，不替换模型源码。设置 `PYTHONDONTWRITEBYTECODE=1`，`PYTHONPATH` 只指向临时归档根目录。这里的“引用”不是操作系统级只读挂载，依赖版本仍由原重建入口核验。
5. 执行归档内的 `scripts/eval_from_source.py`，保留其完整 config 与 checkpoint 校验。Composer 编译的 scene/task 仍在另一个临时目录；回放结束后临时源码及编译结果删除，输出 JSON／视频保留在用户指定目录。

已有 `shadow_sync_20260925.tar.gz` 未包含预览脚本：可用性按模式检查；请求 render／preview 时会报告具体缺失路径。当前完整归档为 `before_preparation_20260926.tar.gz`，不是将旧包源码与当前脚本拼接。

## 实际验收

- `training_mjlab/.venv/bin/python -m pytest tests/test_eval_source_archive.py -q`：**6 passed**。覆盖路径穿越、绝对路径、符号链接、环境文件注入、额外源码文件导致散列集合不符、缺失预览脚本。
- 用完整归档实际回放旧 5M 的 `final.pt`：验证 **160 个源码／资源文件**，成功重建临时场景，完整评估 158 步，F1 **0.9405063391**，与原训练 final 日志一致；return **521.16058**，原日志 **521.16052**，不声称浮点轨迹逐 bit 相同。
- 退出后验证本次临时归档目录不存在。没有修改旧 checkpoint 或 config，没有产生持久冻结 scene/task。
- 用缺预览脚本的旧归档执行 `--mode render`：预期返回码 **2**，明确报告缺 `preview/scripts/render_mjlab_checkpoint.py`；没有调用当前预览脚本，也没有创建输出视频目录。

证据在 `preview/outputs/diagnostics/speed_20260926/`：`archive_final_evaluation.json`、`archive_final_evaluation.log`、`archive_missing_preview.log`。相关速度审计见 `docs/training_speed_audit_20260926.md`。

## 默认视频导出入口的兼容修复

`preview/scripts/export_latest_training_video.sh` 原来默认选择旧 Shadow 5M run，再调用当前源码的 `eval.sh`。本轮源码更新后，这会按设计触发散列不符。现仅对规范路径为 `training_mjlab/outputs/sac-twinkle-shadow-sync-5m-20260925-171119-593333` 的历史 run 使用上述匹配归档；用户传入其他 run 时继续使用正常 `eval.sh`，仍要求当前源码匹配，不能自动跳过校验。输出保留纳秒时间戳的唯一目录。

检查点实际上由训练器先写临时文件、`fsync` 后 `os.replace` 原子发布。已修正“训练器直接写最终文件”的旧注释：选择器只扫描最终 `.pt` 名称，排除临时保存文件；保留非空／文件年龄保护，回放时仍验证所选检查点和完整元数据。

第一次实际渲染发现归档外的 SoundFont／FluidSynth 依赖缺失，错误为 `Piano SoundFont missing`；因此补上前述 `audio_runtime` 可选引用。没有改归档源码或关闭音频。失败日志保留在 `preview/outputs/diagnostics/speed_20260926/default_export_verification.log`。

修复后实际执行默认命令成功：

```bash
bash preview/scripts/export_latest_training_video.sh
```

输出：`preview/outputs/replays/shadow_sync_final_20260926-004021-833751768/replay.mp4`。报告为同目录 `report.json`，渲染 F1=0.9405063391、return=521.16052246、158 个策略步，与旧 final 原训练评估相符。`ffprobe` 确认 **1280×720 H.264＋AAC，8.95 秒**，179 视频帧（包含原先 0.05 秒初始帧与 1 秒结尾停留）；这是旧模型视频，未加入新准备动作。音频由实际琴键／踏板状态生成，SoundFont 散列已记录。

`bash -n` 通过，归档入口的 6 项测试再次通过；没有改训练包模块。成功入口日志为 `preview/outputs/diagnostics/speed_20260926/default_export_success.log`。本次完整视频验证了默认历史 run 分支；其他 run 保持原有调用路径，没有重复执行其他长期训练的完整导出。
