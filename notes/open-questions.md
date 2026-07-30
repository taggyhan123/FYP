# Open questions and implementation risks

1. **Representative traces:** ToolRet relevance and BFCL menus are not
   chronological production calls. Which approved SoC trace can provide
   successful-call and session-order evidence?
2. **Template exactness:** Which Qwen model revision and vLLM chat template will
   be frozen? Tool order at the API level is insufficient unless rendered token
   IDs are also preserved.
3. **Quality sensitivity:** Does reordering alter tool selection, arguments, or
   no-tool behavior? BFCL evaluation must accompany every systems result.
4. **Cache budget:** The local trie assumes unlimited retained nodes. What GPU KV
   budget and eviction policy reflect the intended deployment?
5. **Block boundaries:** Tool-unit token sums approximate but do not equal token
   counts after list delimiters and chat templates. Cluster analysis must use the
   complete rendered prompt and actual vLLM block size.
6. **Frequency semantics:** Which experiments use gold requirement, exposed
   menu, retrieved menu, successful call, or synthetic replay frequency? Tables
   must label this dimension.
7. **FP-tree extension:** Is conditional co-occurrence strong enough to justify
   path-specific ordering beyond global support?
8. **Authorization:** If later work retains inactive schemas in context, how
   will a fresh active-tool manifest and constrained validation prevent calls to
   inactive or unauthorized tools?
9. **Versioned metrics:** Which vLLM version is available on the cluster, and
   which Prometheus names are exposed there? The probe tolerates missing names,
   but the final report must state the exact version.
10. **Novelty boundary:** CacheWeaver and ContextPilot already study
    cache-aware ordering. The defensible TATM contribution should be specific to
    tool schemas: schema cost, workflow co-occurrence, active authorization,
    function-call quality, and cache admission.
