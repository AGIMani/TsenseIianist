# 2026-10-01 停止训练、保留资源及当前视频

用户要求导出当前视频，中断所有训练，但不释放卡。

北京时间18:16左右，停止H100追加等待器PID3971706及原队列协调器PID3956692，再取消训练步骤396.6；A100取消训练步骤1364.10。只取消子步骤，未取消预留作业。H100两份runtime已标记stopped_by_user，R0标记interrupted_by_user，尚未执行的R1/R2/R3/R4_gamma08标记cancelled_by_user，避免后续自动开训。

资源预留保持RUNNING：A100作业1364，两卡，原到期时间北京时间10月3日11:50:28；H100作业396，两卡，原到期时间10月3日13:04:35。未延长到期时间。

采用停止前最新完整存档：A100 step-11000.pt（原运行最后日志11400步，5836800样本）；H100 R0 step-6000.pt（最后日志6200步，3174400样本）。未将尚未存档的内存状态声称为已保存，视频也不是best.pt回放。

## 导出与兼容处理

两边均通过scripts/hpc/submit.sh --allocation，在原预留内执行replay --mode render，正面手部视角、完整曲目、实际物理触键音频，分别使用1364/steps/10/source.tar.gz及396/steps/6/source.tar.gz；保留166文件源码散列检查。

首次步骤1364.11/396.7在weights_only加载时拒绝评估记录中的numpy.float64。使用日志目录prepare_video_checkpoint.py，仅白名单加载已知NumPy标量/类型，在新导出副本中将NumPy标量转为等值Python标量。两边所有187个张量逐项torch.equal通过，所有其他值递归相等；新副本无需白名单即可weights_only=True读取。原PT/config及训练源码不修改。原/副本SHA和变更字段保存于inputs_a100、inputs_h100的conversion.json。

第二次导出步骤1364.12完成A100视频：40fps、2678帧、66.95秒，覆盖2637策略帧/26370物理子步，实际触键音频非静音，完整FFmpeg解码退出0。输出preview/outputs/stopped_training_20261001/a100_step11000/replay.mp4。

H100步骤396.8发现FluidSynth依赖libXss.so.1缺失。包下载源返回502，改为将A100已安装Ubuntu libxss1 1:1.2.3-1build2的库复制到H100用户audio_runtime/root/usr/lib/x86_64-linux-gnu；未改系统库。两边SHA256一致：2c30b5f8ae4a1a733c70385229fe2143de270f292097a61879f4696abe63545b。步骤396.9已完成带音频渲染，原始输出保存在H100数据盘outputs/stopped_training_20261001/h100_step6000。网络传输过慢，停止整目录传输，删除本地未完成的played.wav副本；保留远端全部原文件。用CPU FFmpeg将视频另编码为libx264 CRF28、音频直接复制的replay_share.mp4（5439060字节），同分辨率/帧率/完整时长。将分享视频、report、preview和audio_events复制到当前项目preview/outputs/stopped_training_20261001/h100_step6000。

两份原始视频均40fps/2678帧/66.95秒，实际音频覆盖26370物理子步，音频非静音且完整FFmpeg解码退出0。H100原视频SHA256为0661cd512047527d7f429c6cbb3a05e6db391fe7076b19392fe3ae81f1840ceb；分享版SHA256为3196125ff9cff3d63ef9b3bfadc8ace71e495b3742fbc605607c6db754341e7a。最终两边仅剩预留batch步骤，未发现rl_torch.train进程；H100两协调器PID已不存在。停止清单快照保存在logs/training_stop_20261001/。

## 限制

这是停止时已有检查点的新回放；浮点接触分叉可能造成与训练评估不同的轨迹。尚未完成的R0/其他对照不能作为完整同预算结果。NumPy评估标量的训练保存端修复留待后续，此次仅制作等值导出副本。
