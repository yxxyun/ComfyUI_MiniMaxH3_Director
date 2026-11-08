# 「全部导出」合并音轨的段间清理

分段生成时每段音频由 AV latent 独立解码，各段自带响度和各自的低频层。`torch.cat` 直拼会在每个段界留下电平台阶，并在整条时间轴下垫一层持续的隆隆。实测一条 12 段 / 67.5 s 的导出：段中位 RMS 跨度 23.6 dB，11 个段界里 6 个台阶超过 6 dB。

## 处理内容

`_merge_generated_segment_audios()` 在拼接后调用 `director/audio_postprocess.py`：

| 步骤 | 做法 | 参数 |
|---|---|---|
| 段级电平匹配 | 每段按中位帧 RMS 向全体中位数靠拢，单段限幅 ±9 dB | `MAX_GAIN_DB` |
| 段界过渡 | 以段界为中心 ±80 ms 做 dB 域线性斜坡，只缩放不改波形 | `BOUNDARY_RAMP_SEC` |
| 高通 | 二阶高通去掉 20-80 Hz 的 audio VAE 泄漏 | `HIGHPASS_HZ` |
| 峰值 | 整体缩放，最响样本落在 −2 dBFS，给 AAC 留过冲余量 | `PEAK_CEILING_DBFS` |

## 不使用交叉淡化

段界波形本身是样本连续的（相邻样本差 ~0.01，比全片 99.9 分位差低两个数量级），重叠两侧只会在每个段界挖出 −20 ~ −47 dB 的凹陷，听感变成每段一次「抽气」。电平用增益斜坡处理，样本之间永不叠加。

## 关闭

timeline `output.audioLevelMatch` 设为 `false` 只关电平匹配，高通和峰值限制仍生效。老工作流没有该字段，默认开启。

## 验收

同一条 12 段素材重跑：段间中位极差 23.65 → 15.09 dB，最静帧 p5 −43.7 → −54.3 dBFS，20-80 Hz 能量 −43.0 → −55.5 dB，对话/底噪动态范围 31.1 → 35.4 dB，超满刻度样本 46 → 0，段界 ±5 ms 最低电平 −47.57 → −48.07 dB（未挖凹陷）。

测试：`python -m unittest discover -s tests -p "test_audio_postprocess.py"`（纯 torch，不需要 FFmpeg）。
