[English](trace-schema.md) | **简体中文**

# Claude Code 会话记录格式：观察到的性质

从 `~/.claude/projects/` 下面的真实会话记录逆向出来的，不是从文档。这里每一条说法，在采集器
依赖它之前都对着真实文件核过。会话记录是只读的。

这些是从**一台机器**的语料归纳出来的，而这是本项目最大的已知未知（见 README 的「最大的已知
风险：n = 1」）。那份语料有多大，这里**故意不写**：一个体量描述的是那台机器，不是这个格式。
读者需要的是「某条性质有没有反例」，而 `./bin/agent-profiler verify` 会拿你自己的语料重新核
一遍每一条，这比让你相信别人机器上的一个数字要好。

## 观察到的记录类型

`assistant`、`user`、`api-request`、`api-request-blob`、`api-request-shape`、`attachment`、
`atis-latch`、`last-prompt`、`ai-title`、`queue-operation`、`file-history-snapshot`、
`file-history-delta`、`cost-state`、`system`、`mode`、`tools-used`、`permission-mode`。

这个集合之外的任何东西都会报进 `integrity.unknown_record_types`，而不是丢掉——这样宿主程序
升级就表现为可见的漂移，而不是悄悄少算。`permission-mode` 就是这么发现的：某次运行把它报成
了未知类型，而不是把它扔了。

## 性质 1 —— 一次回复会被写成好几条记录

一次包含多个内容块的 API 回复，会被**每块写成一条记录**。这些记录共享 `message.id` 和
`requestId`，靠 `apiBlockIndex` 区分，而且都**重复了 `message.usage`** —— 一般情况下完全相同；
如果这次回复是边生成边记下来的，那它们是递增到一个最终值。

每一个检查过的会话里，不同 `message.id` 的个数都少于带用量的记录条数。重复的用量载体是常态，
不是边缘情况。

```
<message id>  blockIdx=0  blocks=[text]      in=2  cache_read=56285  out=186
<message id>  blockIdx=1  blocks=[tool_use]  in=2  cache_read=56285  out=186
```

**后果：** 用量必须按 `message.id` 只求和一次。按记录求和把某个会话的 token 数抬高了约 44%。

**该用 `message.id` 做键，不是 `requestId`。** 这一点是专门核过的：某个会话里的消息全都共享
`requestId: null`，所以按 `requestId` 去重会把它们携带的 token 全部丢掉。

**但「取第一次出现」并不是无损的。** 这份文档以前写着「从没观察到同一个 message id 携带两个不同
的用量值」，把机器上每一份会话都扫过一遍之后，这句被推翻了：有些回复是**边生成边记**下来的，所以
那些副本带的 `output_tokens` 是递增的 —— 先是一个很小的数，然后大一些，最后才是真实值。取第一条
就等于报了最小的那个。这些副本并不互相矛盾，只有一条是最终的；而最终那条认得出来，因为它的用量里
带着 `iterations` 和 `output_tokens_details` 这两个完成字段，中途的副本没有。

**后果：** 取的是**已定稿**那条，而不是第一条；如果没有任何一条带标记，才退回取第一条。`verify`
现在核的是两件事，不是一件：这些副本彼此不矛盾（数值变小，或者有两条都带完成标记，就意味着两条都
不可信），以及确实递增时最后那条带标记。两条在这份语料上都成立；边生成边记的形态在这里很少见，它
造成的误差也只有输出 token 的千分之几 —— 但这个误差会随着「harness 有多依赖流式输出」而变大，
而那是构建版本的性质，不是机器的性质。

## 性质 2 —— `api-request-blob` 是请求的回放

`api-request-blob` 记录重复了发出去的请求体，其中包含助手的 `tool_use` 块。在所有记录里天真
地扫一遍 `tool_use`，每次调用都会被找到两遍。

核实方式是在一个会话内部比较两个集合：`assistant` 记录里的每一个 `tool_use` id，都在
`api-request-blob` 记录里再次出现——完全重叠，并且没有任何只有回放才带的 id。

**后果：** 工具调用只从 `assistant` 记录里统计，并按 `tool_use` id 去重。回放仍然保留着，作为
「究竟发出去了什么」的原始证据。

## 性质 3 —— 延迟是通过 `requestRef` 关联的，不是 `requestId`

`assistant.requestRef` 等于 `api-request.id`。`assistant.requestId` 是另一个标识符，在
`api-request` 里什么都匹配不上。

```
api-request.id       = <uuid A>
assistant.requestRef = <uuid A>   <- links
assistant.requestId  = <uuid B>   <- does not
```

按 `requestId` 匹配一条都关联不上；按 `requestRef` 每一条请求都关联上了。

## 性质 4 —— `cost-state` 是宿主程序自己的账

新版本会话的最后一条记录。带有 `totalCostUSD`、`totalAPIDuration`、`totalToolDuration`、
`totalLinesAdded/Removed`、`startTime`，以及按模型分的 `modelUsage`。这是一个可以用来校验采集
器的**独立来源**，`cross_check` 就是为此存在的。

两本账合法地出现差异，有两个真实原因：

- **续跑的会话。** `cost-state.startTime` 可以**晚于**第一条记录，于是会话记录里有宿主程序从
  未计费的历史。在一个会话上发现：它的记录比计费开始早了约 47 分钟，比值落在 2.1x–15x。归类
  为 `resumed_session_superset`，不是告警。
- **被打断的回复。** `stop_reason: null` 的回复会被记录下来，但被排除在宿主程序的成本核算之外。
  在一个会话上发现：一条被打断的回复携带的缓存读取 token **恰好**等于那笔差额，而它的
  `output_tokens: 8`。这类回复计入 `observed`（会话记录确实记了它们），排除在 `cross_check`
  之外（宿主程序确实不为它们计费），这样比较才是同口径的。

压缩和裁剪的方向相反：宿主程序的总数留着，记录没了，所以采到的**更少**是正常的。只有采到
**更多**才说明去重有缺陷，这正是 `double_counting_suspected` 守的东西。告警线设在 +2%，高于真
实会话上看到的约 1% 记账波动，又远低于一条重复记录会造成的量级。

语料里的每一个会话：**0 个 `double_counting_suspected`，0 个无法解释的正向偏差。** 每一处分歧
都有归因。

## 覆盖边界：宿主程序版本

`cost-state` 和 `api-request` 在更老的会话记录里不存在。缺这两样的会话全部来自 2.1.2xx 版本；
2.1.3xx 的会话两样都有——除了扫描时仍然开着的会话，它们还没写最后那条记录。

所以覆盖率是会话的性质，不是工具的性质：

| 能力 | 在哪些会话里可得 |
| --- | --- |
| token、工具、轮次、错误 | 每个会话 |
| 独立交叉校验 | 写过 `cost-state` 的会话 |
| 单次请求延迟 | 写过 `api-request` 的会话 |

`report` 会针对它面前这份语料给出比例。在这里写一个比例，描述的只是某一台机器的升级史，别无
其他。

## 没有观察到的

整份语料里不存在任何 `isSidechain: true` 的记录，所以子 agent 的轨迹是**未验证**的。采集器会
统计 `sidechain_records`，但没有任何真实数据跑到过它；`Task` 派生出来的工作可能写在别处，也可
能根本没写。这是一个已知缺口，不是一个受支持的功能。
