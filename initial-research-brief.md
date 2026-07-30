# Initial Research Brief

## Efficient Trie-Aware Tool Memory for Dynamic Tool-Using LLM Agents

**Working name:** TATM (Trie-Aware Tool Memory)  
**Project type:** Publication-oriented undergraduate research project / FYP  
**Research areas:** LLM agents, tool use, agent memory, prefix caching, KV-cache reuse, inference systems

---

## 1. Project Motivation

Modern LLM agents may connect to many MCP servers and expose hundreds or thousands of tools. Each tool definition contains a name, description, input schema, constraints, and sometimes examples. Loading a large tool catalog into every prompt can consume a substantial part of the context window and increase prefill latency, KV-cache memory usage, and decoding cost.

Dynamic tool search addresses part of this problem by retrieving only a small set of tools for each request. However, different requests may retrieve different tool sets or the same tools in different orders. This makes exact prefix-cache reuse difficult for the selected tool definitions.

This project studies tool definitions as a form of **persistent, workload-aware agent memory**. The initial goal is to understand whether tool usage has enough frequency skew, co-occurrence structure, and workflow locality to support a trie-aware organization of tool definitions. The longer-term goal is to combine exact prefix memory for frequent tool paths with dynamic memory for long-tail tools.

The project is planned toward a publishable research system. However, the first tasks deliberately focus on exact, measurable, and low-risk components so that the student can establish a reliable experimental foundation before modifying KV-cache internals.

---

## 2. High-Level Research Questions

The project will initially investigate the following questions:

1. **Tool locality:** Do tool requirements and successful calls exhibit hotspots, co-occurrence patterns, repeated workflows, or session locality?
2. **Prefix organization:** Can deterministic, cost-aware tool ordering create longer reusable prefixes than the original, alphabetical, or frequency-only order?
3. **Trie-aware memory:** Can frequently occurring tool sequences be represented as a weighted trie or prefix memory under a limited cache budget?
4. **Retention trade-off:** Is it sometimes beneficial to retain a small number of cached but currently inactive tools in order to reuse a longer prefix?
5. **Long-tail tools:** For tools not covered by the exact prefix memory, when is text prefill preferable to loading or composing a precomputed KV representation?
6. **Quality and safety:** How do additional or reordered tools affect tool selection, argument generation, no-tool decisions, and unauthorized-tool prevention?

The first three questions define the initial implementation scope. The later questions are research extensions that may be pursued once the basic measurements and baselines are reliable.

---

## 3. Required Background and Reading

### 3.1 Core foundations

| Topic | Resource | What to understand |
|---|---|---|
| Transformer attention | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Self-attention, causal attention, prompt processing, and autoregressive decoding |
| Rotary position encoding | [RoFormer](https://arxiv.org/abs/2104.09864) | How RoPE encodes relative position in queries and keys |
| KV-cache memory management | [PagedAttention / vLLM paper](https://arxiv.org/abs/2309.06180) | Why KV cache is paged and how blocks are shared |
| Exact prefix caching | [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | Exact token-prefix reuse, cache hits, and limitations |
| Trie-based KV reuse | [SGLang paper](https://arxiv.org/abs/2312.07104) | RadixAttention and reuse of shared prefixes |
| Frequent-pattern tries | [FP-Growth paper](https://www.cs.sfu.ca/~jpei/publications/sigmod00.pdf) | Frequency-ordered items, prefix sharing, and frequent pattern mining |

### 3.2 MCP and tool-use background

| Topic | Resource | What to understand |
|---|---|---|
| MCP overview | [Official MCP introduction](https://modelcontextprotocol.io/docs/getting-started/intro) | MCP clients, servers, tools, resources, and prompts |
| MCP tool schema | [Official tools specification](https://modelcontextprotocol.io/specification/draft/server/tools) | Tool names, descriptions, JSON Schema inputs, outputs, and errors |
| Large tool-definition overhead | [Anthropic: Advanced Tool Use](https://www.anthropic.com/engineering/advanced-tool-use) | Why large tool catalogs are expensive and how Tool Search works |
| Function calling evaluation | [BFCL repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) | Tool-name and argument evaluation formats |

### 3.3 Related research to read after the foundations

| Work | Link | Relevance to this project |
|---|---|---|
| Prompt Cache | [Paper](https://arxiv.org/abs/2311.04934) | Modular reuse of precomputed attention states |
| CacheBlend | [Paper](https://arxiv.org/abs/2405.16444) | Why non-prefix KV blocks cannot always be directly combined |
| ContextPilot | [Paper](https://arxiv.org/abs/2511.03475) · [Code](https://github.com/EfficientContext/ContextPilot) | Cache-aware context ordering and de-duplication |
| CacheWeaver | [Paper](https://arxiv.org/abs/2606.19667) | Prefix-tree-based ordering of overlapping retrieved evidence |
| ToolRet | [Paper](https://arxiv.org/abs/2503.01763) · [Code](https://github.com/mangopy/benchmarking-tool-retrieval) | Large-scale tool retrieval and tool-catalog construction |
| MCPAgentBench | [Paper](https://arxiv.org/abs/2512.24565) | MCP-style tasks with distractor tools and multi-step execution |
| MCP-Atlas | [Paper](https://arxiv.org/abs/2602.00933) | Real MCP servers and cross-server multi-step workflows |

The student is not expected to fully understand every advanced paper before beginning. The first reading goal is to understand the problem setting, the exact-prefix baseline, and the difference between text-level ordering and KV-level composition.

---

## 4. Initial Datasets and Benchmarks

### 4.1 ToolRet — primary dataset for large tool catalogs

**Resources:**  
[Paper](https://arxiv.org/abs/2503.01763) · [Repository](https://github.com/mangopy/benchmarking-tool-retrieval) · [Tool corpus](https://huggingface.co/datasets/mangopy/ToolRet-Tools)

ToolRet contains thousands of retrieval tasks and a large corpus of tool definitions. It is the main starting point for:

- inspecting schema lengths and tool-description structure;
- building large candidate catalogs;
- evaluating tool retrieval;
- generating distractor tool sets;
- studying whether tools form identifiable domains or clusters.

**Important limitation:** ToolRet is a retrieval benchmark, not a production request log. Its task frequency should not automatically be interpreted as real-world tool popularity.

### 4.2 BFCL V4 — primary benchmark for function-call correctness

**Resources:**  
[Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) · [Repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)

BFCL should be used to evaluate:

- correct tool selection;
- argument structure and values;
- multiple and parallel tool calls;
- irrelevant/no-tool cases;
- format validity and executable function calls.

BFCL provides task-level gold functions, but it is also not a natural temporal access trace. It is suitable for quality evaluation and for constructing controlled tool-menu workloads.

### 4.3 MCPAgentBench — later-stage reproducible MCP evaluation

**Resource:** [Paper](https://arxiv.org/abs/2512.24565)

MCPAgentBench uses real-world MCP-style definitions with simulated tools and distractor candidates. It is useful after the initial baseline is stable because it supports multi-step tasks and repeatable execution without relying entirely on live external services.

### 4.4 MCP-Atlas and MCP-Bench — later-stage realism checks

**Resources:**  
[MCP-Atlas paper](https://arxiv.org/abs/2602.00933) · [MCP-Bench repository](https://github.com/Accenture/mcp-bench)

These resources are useful for evaluating real or realistic MCP workflows, cross-tool dependencies, and end-to-end task completion. They should not be the first implementation target because setup and execution are more complex.

### 4.5 Trace-construction requirement

The public benchmarks above generally provide tasks, tool definitions, relevance labels, or execution trajectories. They do not necessarily provide a representative chronological production trace.

The student must therefore distinguish between:

- **benchmark frequency:** how often a tool appears in a dataset;
- **gold requirement frequency:** how often a task requires a tool;
- **retrieved/exposed frequency:** how often a retriever places a tool in the menu;
- **successful-call frequency:** how often a tool appears in successful execution;
- **synthetic workload frequency:** a controlled replay distribution used for systems experiments.

Initial experiments should include at least:

- the empirical benchmark order;
- a uniform replay;
- one or more skewed replays;
- a session-bursty replay where related workflows occur near each other.

The purpose is not to assume a Zipf distribution, but to test how the proposed memory behaves under different levels of locality.

---

## 5. Reference Code and Systems

| Project | Link | Suggested use |
|---|---|---|
| vLLM | [GitHub](https://github.com/vllm-project/vllm) | Main serving engine and exact prefix-cache baseline |
| vLLM APC documentation | [Documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) | Reproduce cache hits and inspect limitations |
| SGLang | [GitHub](https://github.com/sgl-project/sglang) | Reference for radix-tree KV reuse |
| ToolRet | [GitHub](https://github.com/mangopy/benchmarking-tool-retrieval) | Tool corpus, retrieval tasks, and evaluation |
| Gorilla / BFCL | [GitHub](https://github.com/ShishirPatil/gorilla) | Function-calling data and checkers |
| MCP Python SDK | [GitHub](https://github.com/modelcontextprotocol/python-sdk) | Parsing and running MCP tools |
| LMCache | [GitHub](https://github.com/LMCache/LMCache) | Later reference for external KV-cache storage |
| XGrammar | [GitHub](https://github.com/mlc-ai/xgrammar) | Later reference for constrained tool-call generation |
| ContextPilot | [GitHub](https://github.com/EfficientContext/ContextPilot) | Reference for cache-aware context ordering |

The first implementation should use the public APIs and metrics exposed by vLLM whenever possible. Modifying CUDA kernels or attention internals is not an initial task.

---

## 6. Initial Hands-On Tasks

### Task A — Prepare a short reading note

Write a concise note that explains:

- how tool definitions enter an LLM prompt;
- the difference between prefill and decode;
- what is stored in the KV cache;
- when exact prefix caching can and cannot be reused;
- why changing tool order changes cache reuse;
- why deleting or concatenating arbitrary KV blocks is not necessarily exact;
- how tries or FP-trees may represent repeated tool sequences.

The note should include at least one self-drawn diagram of the proposed request flow.

### Task B — Reproduce exact prefix caching

Set up a small instruct model with vLLM and verify:

1. two identical prompts reuse the prefix;
2. changing one tool definition causes a cache miss after the changed point;
3. changing the order of two tools changes the reusable prefix;
4. cache-enabled and cache-disabled generation produce the same result for the same token sequence;
5. the experiment records prompt length, cached tokens, prefill latency, TTFT, and GPU memory.

The goal is to build a trustworthy measurement harness before testing new algorithms.

### Task C — Inspect and normalize tool datasets

For ToolRet and a manageable BFCL subset:

- document the available fields and file formats;
- parse tool names, descriptions, parameters, required fields, and examples;
- implement one canonical serialization format;
- measure token length per tool;
- report missing or malformed schema fields;
- group tools by source, domain, or server where possible.

The output should be a reproducible script and a short data report.

### Task D — Analyze initial tool-access patterns

Using gold labels or successful trajectories where available, compute:

- tool occurrence frequency;
- schema-token-weighted frequency;
- pair and triple co-occurrence;
- conditional transitions between tools;
- number of tools per task;
- total schema tokens per task;
- domain-level locality;
- trie size and compression under different orderings.

Compare at least:

- original order;
- alphabetical order;
- random order with fixed seeds;
- frequency order;
- schema-cost-weighted frequency order;
- FP-tree-style global order.

The student should clearly state which statistics come directly from the benchmark and which are generated through a replay model.

### Task E — Build the first exact ToolTrie baseline

Implement a prompt-level prototype that:

- receives a selected tool set;
- applies a deterministic global or trie-aware order;
- serializes the tools consistently;
- sends requests through vLLM with prefix caching enabled;
- records exact prefix reuse and latency;
- falls back to normal selected-tool text prefill without changing model semantics.

This first baseline must not modify KV tensors. It should establish whether tool ordering and repeated workflows can create useful prefix-cache reuse.

### Task F — Produce an initial research report

The first report should contain:

- dataset inventory and limitations;
- schema-length and tool-frequency figures;
- co-occurrence or workflow analysis;
- prefix-cache sanity results;
- comparison of simple tool orderings;
- observed bottlenecks;
- a recommendation on which research extension is best supported by the data.

The recommendation may support, reject, or refine the initial hypothesis. Negative findings are acceptable if they are carefully measured.

---

## 7. Initial Experimental Questions

The initial experiments should answer the following:

1. How much of TTFT is caused by selected tool-schema prefill?
2. How much exact prefix reuse is available without changing the tool set?
3. Does frequency-based ordering improve reusable prefix length?
4. Does weighting frequency by schema length or measured prefill time perform better?
5. How much additional benefit comes from pair/triple workflow structure?
6. How sensitive are results to request ordering and session locality?
7. Does tool reordering change function-call accuracy?
8. Under which workloads does trie-aware ordering provide little or no benefit?

The project should report both positive and negative regimes. A useful system should know when to use the optimization and when to fall back to ordinary Tool Search plus text prefill.

---

## 8. Desired Initial Outcomes

The initial stage should produce:

- a clean and reproducible tool-schema processing pipeline;
- a reliable vLLM prefix-cache benchmark;
- an evidence-based characterization of tool locality and cacheability;
- a set of exact, quality-preserving ordering baselines;
- a first ToolTrie prototype using native prefix caching;
- an experimental basis for deciding whether to pursue cached-inactive retention, dynamic long-tail KV memory, or another direction.

The primary success criterion is not a predetermined speedup number. It is obtaining a trustworthy answer to whether public tool workloads contain exploitable prefix locality and identifying the conditions under which the optimization is beneficial.

---

## 9. Possible Research Extensions

After the initial work is complete, promising extensions may include:

### 9.1 Budgeted cached-inactive retention

Allow the planner to keep a small cached tool that is not required by the current request when this enables a longer prefix hit. This requires careful evaluation of context overhead, decode KV I/O, tool confusion, and permissions.

### 9.2 Active-tool manifest and constrained generation

Separate the tools present in cached context from the tools that may actually be called. A fresh manifest and constrained decoder can restrict generation to the active and authorized tool set.

### 9.3 Cache placement and admission

Select which trie nodes, tool bundles, or tool schemas should remain in GPU memory under a limited cache budget using measured reuse value rather than frequency alone.

### 9.4 Long-tail dynamic KV memory

Independently precompute long tool definitions and investigate whether their KV state can be safely relocated or linked after an exact cached prefix. This is an advanced extension and must always retain text prefill as the quality-preserving fallback.

### 9.5 Integrated memory planner

Jointly choose the active tool set, exact cached prefix, optional retained tools, and materialization method for residual long-tail tools.

These extensions define the publication-oriented direction, but the choice among them should be guided by the initial data and experiments rather than fixed in advance.

---

## 10. Working Principles

- Start with exact and reproducible baselines before introducing approximate KV reuse.
- Separate retrieval errors from caching or serving errors by evaluating both gold-tool and retrieved-tool modes.
- Never interpret benchmark frequency as production popularity without stating the assumption.
- Preserve canonical tool serialization across all experiments.
- Report accuracy, latency, cached tokens, active context length, and memory together.
- Keep ordinary selected-tool text prefill as a fallback throughout the project.
- Document negative results and crossover regimes; they are important research findings.
- Prefer small, reviewable experiments over large uncontrolled runs.

---

## 11. First Discussion Deliverables

Before the first technical design review, prepare:

1. the reading note from Task A;
2. a working vLLM prefix-cache sanity script;
3. a dataset inventory for ToolRet and BFCL;
4. a preliminary schema-length and frequency analysis;
5. a one-page proposal for the first exact ToolTrie baseline;
6. a list of unresolved questions or implementation risks.

The project direction will be refined after reviewing these results.
