# Principles · 原则

Both languages live in one file here. The list is short enough that two files would add a
way for them to drift without adding anything for a reader.

两种语言放在同一个文件里。这份清单足够短，分成两个文件只会多一条走偏的路，对读者没有任何
好处。

1. **Read-only by default and by design.**
   默认只读，而且是设计上的只读。
2. **Evidence before recommendation.**
   先有证据，再给建议。
3. **Observed waste and estimated waste must never be conflated.**
   观测到的浪费和估算出的浪费，绝不混为一谈。
4. **File size, component count, or low usage alone do not prove waste.**
   文件大小、组件数量、使用次数少，单独都不能证明浪费。
5. **Deterministic analysis should replace LLM judgment whenever it can do so reliably.**
   只要确定性分析能可靠地胜任，就用它替掉让模型来判断。
6. **Profiler overhead must eventually be measured and reported.**
   工具自身的开销最终必须被测量并报出来。
7. **No overall 0-100 score in v0.1; first make the measurements trustworthy.**
   v0.1 不做 0-100 总分；先把测量做到可信。
