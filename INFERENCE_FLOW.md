# Octo Inference Flow

这份说明对应当前仓库里的 Octo 推理主链路，重点结合 ALOHA 这个 finetune 示例来理解。

## 1. 推理入口

通常推理会经过这几个函数：

- `OctoModel.load_pretrained(...)`
- `model.create_tasks(...)`
- `model.sample_actions(...)`

相关代码：

- [octo/model/octo_model.py](/home/cjt/octo/octo/model/octo_model.py)

## 2. 模型先加载什么

`load_pretrained(...)` 会加载：

- `config.json`
- `example_batch.msgpack`
- `dataset_statistics.json`
- checkpoint 参数

作用：

- 用 `config` 重建模型结构
- 用 `example_batch` 约束输入格式
- 用 `dataset_statistics` 做动作反归一化

## 3. 输入是什么

在当前的 ALOHA finetune 示例里，输入不是视频文件，而是按时间组织的轨迹数据。

实际用到的输入有：

- `top`：主相机图像
- `state`：低维状态
- `language_instruction`：语言指令

对应代码：

- [examples/02_finetune_new_observation_action.py](/home/cjt/octo/examples/02_finetune_new_observation_action.py#L73)

这个示例里 `window_size=1`，所以更接近：

- 单帧图像
- 加上 proprio
- 加上语言
- 预测未来动作序列

不是传统意义上的整段视频输入。

## 4. task 是怎么构造的

`create_tasks(...)` 会把文本或 goal 图像整理成统一的 `task dict`。

如果只给文本：

- `language_instruction` 会被 tokenizer 编码
- 没提供的 goal 图像会被补零
- 同时生成 `pad_mask_dict`

对应代码：

- [octo/model/octo_model.py](/home/cjt/octo/octo/model/octo_model.py#L76)

## 5. observation 和 task 怎么进入模型

`sample_actions(...)` 会：

1. 检查 `observation` 和 `task` 的 shape
2. 调 `run_transformer(...)`
3. 调动作头 `predict_action(...)`
4. 如果提供了统计量，再把动作反归一化

对应代码：

- [octo/model/octo_model.py](/home/cjt/octo/octo/model/octo_model.py#L174)

## 6. Transformer 里做了什么

`OctoTransformer` 会把输入变成 token 序列：

- task token
- observation token
- readout token

然后送进 block-causal transformer 做融合。

可以粗略理解成：

- 语言先变成 task token
- 图像和 proprio 变成 observation token
- `readout_action` token 专门负责读出动作相关信息

对应代码：

- [octo/model/octo_module.py](/home/cjt/octo/octo/model/octo_module.py#L20)

## 7. ALOHA 这个 finetune 改了什么

这个示例相对原始 Octo 做了三件关键改动：

- 删除 `wrist` 相机输入
- 新增 `proprio` tokenizer
- 把动作头改成 14 维、50 步的动作输出

对应代码：

- [examples/02_finetune_new_observation_action.py](/home/cjt/octo/examples/02_finetune_new_observation_action.py#L109)

所以这个 finetuned 模型的输入输出更像：

- 输入：单相机图像 + 状态 + 语言
- 输出：未来 50 步动作 chunk

## 8. 动作是怎么读出来的

动作头会从 transformer 输出里的 `readout_action` token 读取信息，再变成动作张量。

推理时通常只取最后一个时间步对应的预测，输出形状类似：

- `(batch, action_horizon, action_dim)`

对应代码：

- [octo/model/components/action_heads.py](/home/cjt/octo/octo/model/components/action_heads.py)

## 9. 一句话总结

这个模型的推理流程可以理解为：

- 把图像、状态、语言编码成 token
- 用 transformer 融合
- 从 action readout token 中解码出未来一段动作
- 再把动作映射回真实控制量
