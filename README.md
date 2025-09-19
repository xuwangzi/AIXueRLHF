# AIXueRLHF

A variant of PPO to learn from $\langle s, a, r \rangle$ tuples.

[👉 PPO算法与AIXue算法代码实现对比分析](./PPO算法与AIXue算法代码实现对比分析.md)


## Description

This project implements a variant of Proximal Policy Optimization (PPO) algorithm for reinforcement learning from human feedback (RLHF).

## Features

- PPO variant implementation
- Learning from state-action-reward tuples
- Human feedback integration

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd AIXueRLHF
sh ./scripts/train_example.sh
```

The example dataset is ./data/aixue_test_data with 480 $\langle \text{prompt}, \text{response}, \text{reward} \rangle$ pairs.

You can find the trained model and the logged tensorboard in "your_output_dir".

Don't forget to change the "num_processes" in ./configs/deepspeed_zero3.yaml.

### 测试数据

#### prompt

```python
"你是一名专注于1对1自然拼读教学的英语老师。\n\n# 教学目标\n## 核心课程目标：\n    1. 建立\"字母/字母组合\"与\"发音\"的牢固对应关系（音形联结），掌握单个字母基础发音及常见字母组合发音。\n    2. 发展学生语音意识（音素、音节、重音感知），提升单词记忆与拼写能力。\n    3. 最终目标：实现\"见词能读，听音能写\"。\n## 单节课教学目标：\n    1. 个性化教学：根据学生能力调整内容呈现和练习方式，激发兴趣，提升效果。\n    2. 保持专注与信心：控制单次学习时长，避免连续错误过多，减少疲劳感和挫败感。\n\n# 学生画像\n    1. 年龄：7岁\n    2. 性别：女\n    3. 所在地：中国三线城市\n    4. 英语基础：\n       - 掌握26个英文字母名称。\n       - 仅会极少量简单会话（如：What's your name?）。\n\n# 当前教学状态\n    1. 课节内容：教授字母 A、B、C 的发音（a: /æ/, b: /b/, c: /k/）。\n    2. 当前**环节**：字母 A (/æ/) 的发音练习。\n    3. 主题关联：教学围绕'苹果 (apple)'展开，练习部分如有单词，建议与之相关。\n\n# 教学工具箱 (可选学习范式)\n## 字母教学包含4种基础练习类型（难度递增）, 例如对于字母a：\n1. 纯音素重复：`/æ/ /æ/ /æ/` (重复发音3次)\n2. 音形对应：`a says /æ/` (建立字母与发音关联)\n3. 音素-单词关联：`/æ/ /æ/ apple` (强化发音在单词中的感知)\n4. 综合练习：`a says /æ/, /æ/ /æ/ apple` (整合字母、发音与单词)\n## 智能纠错策略 (根据错误类型选择下一步)\n1. 错误类型A (字母名称错，发音对)：例如学生说'a says /æ/' (a读错，/æ/正确)。  \n    **下一步：** 聚焦字母名称练习。老师示范：'a' (仅字母名称)。\n2. 错误类型B (单词发音错，字母发音对)：例如学生说'/æ/ /æ/ apple' (apple发音错，/æ/正确)。  \n    **下一步：** 聚焦目标单词练习。老师示范：'apple' (仅单词)。\n## 教学控制参数\n1. 单字母(单环节)最大教学次数：4次 (若学生能力强，`综合练习`一次性通过，可减少次数, 直接进入下一个**环节**)。\n2. 单次跟读最大重复次数：1-2次 (避免疲劳)。\n3. 核心原则：及时强化正确，精准纠正错误，保持学习动力。\n\n# 学生学习记录\n1. 学习次数：1 次\n2. 上次练习内容：'/æ/ /æ/ apple'\n3. 上次表现评分：B (部分正确)\n4. 具体错误：单词'apple'中的辅音'/p/'发音有瑕疵。\n\n# 你的任务：制定下一步教学指令\n1. **评估：** 基于教学目标、学生基础、当前环节、可选范式、纠错策略、历史表现及教学控制参数，决定下一步的教学。\n2. 输出格式要求：\n   - 如果结束当前字母教学，直接回复 <END>\n   - 如果需要继续学习，直接回复 下一步跟读的句子"
```

#### response & reward

```python
"下一步跟读的句子：**apple** (仅单词，聚焦单词发音练习，纠正辅音 /p/)", -1
"下一步跟读的句子：  \n**apple**", 0
"下一步跟读的句子：**apple** (仅单词，聚焦单词发音练习，纠正/p/音)", -1
"下一步跟读的句子：  \n**\"apple\"** (仅单词，帮助纠正单词发音错误，属于错误类型B的纠错策略)", -1
"下一步跟读的句子：  \n**\"apple\"** (仅单词，聚焦单词发音练习，纠正辅音 /p/ 的发音瑕疵)", -1
"下一步跟读的句子：  \n**apple**", 0
"下一步跟读的句子：**apple**", 1
"下一步跟读的句子：  \n**\"apple\"** (仅单词，聚焦单词发音练习，纠正/p/音)", -1
```

repeated 120 times

奖励的定义倾向于输出为一整行，并且没有附加信息

### 测试结果

一轮训练之后，输出temperature为0.6的情况下，10次输出均为

```python
"下一步跟读的句子：**apple**"
```

## License

[Add your license here]
