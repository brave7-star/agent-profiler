[English](codex-schema.md) | **简体中文**

# Codex 会话记录格式：观察到的性质

从 `~/.codex/sessions/` 下面真实的 rollout 会话记录逆向出来的，只读。写这份东西是因为：归一化
的事件模型需要**第二个**实现才有价值——只有一个适配器，只能证明一种格式可以被改名。

这些是从一台机器的语料归纳出来的，版本从 0.144.2 到 0.162.0-alpha。和 Claude Code 那份笔记一
样，语料的体量故意不写：它描述的是那台机器，不是这个格式。版本号留着，因为版本是宿主程序的性
质，也是这些说法覆盖范围的一条真实边界。

## 存储与访问

`~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<id>.jsonl.zst`，**zstd 压缩**。这里没有
Python 的 `zstandard` 模块，所以适配器通过 `zstd` 这个可执行文件来管道解压，缺它就抛一个明确
的错误，而不是悄悄什么都不报。

会话是**按日期存的，不是按项目**。`~/.codex/session_index.jsonl` 里只有 `id`、`thread_name`
和 `updated_at`——没有 cwd——所以要把一个会话归到某个项目，必须读它的 `session_meta` 记录。因
此发现阶段只读每个文件开头的若干条记录，一找到 `session_meta` 就停，这样开销与会话数成正比，
而不是与语料体量成正比。

## 记录形状

每一行都是 `{timestamp, ordinal, type, payload}`——一层 Claude Code 没有的外壳。观察到的
`type` / `payload.type` 组合：

| 记录 | 含义 |
| --- | --- |
| `session_meta` | 会话 id、cwd、`cli_version`、`model_provider` |
| `response_item/message` | 一轮；`role` 区分用户和 developer |
| `response_item/custom_tool_call` | 工具调用，`input` 是自由文本 |
| `response_item/function_call` | 工具调用，`arguments` 是 JSON |
| `response_item/custom_tool_call_output` | 结果 |
| `response_item/function_call_output` | 结果 |
| `response_item/reasoning` | 推理内容 |
| `token_usage_record` | **单次回复**的用量，按 `response_id` 索引 |
| `event_msg/token_count` | 会话**累计**用量 |
| `event_msg/item_completed` | `response_item` 条目的重复视图 |
| `turn_context` | 每轮的 cwd、模型、沙箱与审批策略 |
| `compacted` | 上下文压缩，带 `replacement_history` |
| `event_msg/turn_aborted` | 被中止的轮次 |
| `world_state` | 工作区状态快照 |

## 性质 1 —— token 字段的含义不一样

这就是那个让「写第二个适配器」变得值得的发现。两条恒等式在采样到的每一条
`token_usage_record` 上都成立，没有反例：

- `input_tokens + output_tokens == total_tokens`
- `cached_input_tokens <= input_tokens`

也就是说，Codex 的 `input_tokens` **已经包含**了 `cached_input_tokens`。而 Anthropic 的
`input_tokens` **不包含** `cache_read_input_tokens`，后者是单独计费的。

把 `cached_input_tokens` 映射到 `cache_read_input_tokens` 再加到输入数上——这是最顺手的做法
——会把每一个 Codex 输入总数抬高一个缓存量，而缓存量占了其中的绝大部分：在某个根目录上超过
输入 token 的 97%。归一化模型改为陈述不变量：

```
input_fresh + input_cached + input_cache_write == input_total
```

Claude Code：`input_total` 是三个分别计费的量的**和**。
Codex：`input_total` 就是 `input_tokens`，要从它里面**减去**缓存部分才得到 `input_fresh`。

`cache_write_input_tokens` 在**语料里的每一条记录上都是 0**，所以它到底在 `input_tokens` 里面
还是外面是*未验证*的。适配器假定在里面（与 `cached_input_tokens` 一致），并且一旦将来出现一个
非零值与此矛盾，就发出 `negative_fresh_input` 完整性告警，而不是打印一个负数。

## 性质 2 —— 累计值和单次值并排放在一起

三种不同的 token 视图共存，其中两种是累计值：

- `token_usage_record.usage` —— 按回复。**只有这一个被求和。**
- `token_usage_record.turn_token_usage` —— 按轮累计；在采样记录中的绝大多数（约 93%）都超过
  了单次回复的总数。
- `event_msg/token_count.info.total_token_usage` —— 按会话累计。

对 `turn_token_usage` 求和会把总数抬高好几倍。这和 Claude Code 那个重复的 `message.usage` 是
同一类陷阱，形状完全不同。

## 性质 3 —— 没有一个可信的独立总数

Claude Code 的 `cost-state` 提供了一个由宿主程序写的总数，用来校验采集器。Codex 最接近的等价
物——最后那个累计 `token_count`——**对不上**：拿语料里最大的那几个会话，把它和单次回复用量之
和相比，差值在**两个方向上都有**，大到任何容差都吸收不了，而且随压缩次数增长。

与其编一个解释、或者把容差调到刚好通过，这件事被记录为一条限制。**Codex 适配器是靠每条记录的
恒等式 `input + output == total` 来验证的，不是靠一个独立的会话总数**——这确实是一个比 Claude
Code 适配器更弱的保证。每条记录自己声明的 `total_tokens` 都会被核，不一致就抛
`harness_total_mismatch` 告警。

## 性质 4 —— 错误是隐式的

Codex 没有 `is_error` 标志。一次失败的工具调用，只能从输出文本里嵌着的那段 JSON 中的非零
`exit_code` 看出来。适配器会去解析它；输出解析不出来的结果按 `ok` 上报，所以 **Codex 的错误计
数是一个下界**。

## 性质 5 —— 没有延迟数据源

Codex 的记录里没有任何等价于 Claude Code `api-request` 的请求起始时间戳，所以每一个 Codex
`model_response` 的 `latency_seconds` 都是 `null`。在这里报一个推导出来的数字就会超出证据。

## 未验证

- 子 agent / 嵌套轮次的归因。`turn_attribution` 带有 `parent_turn_id` 和 `root_turn_id`，但语
  料里没有任何记录的 `parent_turn_id` 非空，所以「嵌套的 agent 工作被采到了」并未得到证明。
- `event_msg/item_completed` 是**故意**不用的：它是已经消费过的记录的重复视图。
- **`world_state` 曾被错误地当成「一个工作区快照」打发掉。** 它带有 `state.host_skills`，里面
  有三样两个宿主程序中别处都没有的东西：

      usage   -> {"tokens": <int>, "characters": <int>}   这份列表的开销，由宿主
                                                          程序自己测量并记录
      entries -> 每个 skill 的路径，带一个 `shown` 标志
      roots   -> skill 从哪些目录加载

  `usage.tokens` 是 skill 列表的一个**免费、精确**的成本，而项目在别处是要付费跑一对探针来
  *估* 这个数的。只有最近的那些会话带这条记录，所以从它读出来的数字描述的是那些会话、不是整
  段历史——这也是报告为什么会把「语料里有多少带这条记录」印在数字旁边。

  `roots` 暴露出了 `~/.agents/skills`，一个扫描器原本没在看的 skill 目录。`shown` 会被读取，
  但**不**当作使用证据：在采样到的每一条记录里，它对每一个条目都是 `true`，所以它不变化、也
  不携带任何关于「哪个 skill 起了作用」的信息。把一个常量当成测量结果来报，正是这个项目存在
  要避免的错误。
