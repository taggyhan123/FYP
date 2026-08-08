from tatm import vllm_client


def test_parse_prometheus_includes_pressure_metrics() -> None:
    parsed = vllm_client.parse_prometheus(
        """
vllm:num_preemptions_total 3
vllm:kv_cache_usage_perc{model_name="qwen"} 0.91
vllm:num_requests_waiting{model_name="qwen"} 2
vllm:kv_block_idle_before_evict_seconds_count 7
"""
    )
    assert parsed["vllm:num_preemptions"] == 3
    assert parsed["vllm:kv_cache_usage_perc"] == 0.91
    assert parsed["vllm:num_requests_waiting"] == 2
    assert parsed["vllm:kv_block_idle_before_evict_seconds_count"] == 7


def test_tokenize_chat_sends_tools_to_server_renderer(monkeypatch) -> None:
    captured = {}

    def fake_request(method, url, body):
        captured.update({"method": method, "url": url, "body": body})
        return {"tokens": [1, 2, 3], "count": 3, "max_model_len": 4096}

    monkeypatch.setattr(vllm_client, "request_json", fake_request)
    result = vllm_client.tokenize_chat(
        "http://localhost:8000",
        "model",
        [{"role": "user", "content": "hi"}],
        [{"type": "function", "function": {"name": "x"}}],
        chat_template_kwargs={"enable_thinking": False},
    )
    assert result["tokens"] == [1, 2, 3]
    assert captured["url"] == "http://localhost:8000/tokenize"
    assert captured["body"]["tools"][0]["function"]["name"] == "x"
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_require_prefix_cache_reset_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(vllm_client, "reset_prefix_cache", lambda _base_url: False)
    try:
        vllm_client.require_prefix_cache_reset("http://localhost:8000")
    except RuntimeError as error:
        assert "reset failed" in str(error)
    else:
        raise AssertionError("failed reset was accepted")
