from __future__ import annotations

from tatm.prefix_cache_sim import block_hashes, simulate_prefix_cache


def test_only_full_blocks_of_the_shared_prefix_are_cached() -> None:
    first = list(range(40))
    second = list(range(40)) + [999] * 10
    assert simulate_prefix_cache([first, second], [0, 0]) == [0, 32]


def test_divergence_inside_a_block_loses_that_block() -> None:
    first = list(range(48))
    second = list(range(20)) + [999] * 28
    assert simulate_prefix_cache([first, second], [0, 0]) == [0, 16]


def test_exact_repeat_still_computes_one_token() -> None:
    prompt = list(range(48))
    assert simulate_prefix_cache([prompt, prompt], [0, 0]) == [0, 32]


def test_block_hash_depends_on_everything_before_it() -> None:
    left = block_hashes([1] * 16 + [2] * 16)
    right = block_hashes([3] * 16 + [2] * 16)
    assert left[1] != right[1]


def test_finite_cache_evicts_a_request_tail_first() -> None:
    a = list(range(32))                    # blocks A0, A1
    b = list(range(100, 148))              # blocks B0, B1, B2
    a_again = list(range(32)) + [7] * 16
    prompts, completions = [a, b, a_again], [0, 0, 0]
    # Capacity 4: serving B must evict one of A's blocks, and it takes the
    # tail block A1, so A0 survives and the repeat still hits 16 tokens.
    assert simulate_prefix_cache(prompts, completions, capacity_blocks=4) == [0, 0, 16]
    assert simulate_prefix_cache(prompts, completions, capacity_blocks=None) == [0, 0, 32]
