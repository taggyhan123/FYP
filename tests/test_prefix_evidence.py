from tatm.prefix_evidence import (
    RenderedPrefixIndex,
    best_prior_prefix,
    common_prefix_length,
    prefix_pair_evidence,
    token_blocks,
)


def test_common_prefix_length_handles_difference_and_containment() -> None:
    assert common_prefix_length([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_length([1, 2], [1, 2, 3]) == 2
    assert common_prefix_length([], [1]) == 0


def test_token_blocks_marks_only_complete_blocks_cache_eligible() -> None:
    blocks = token_blocks([1, 2, 3, 4, 5], block_size=2)
    assert [block["token_count"] for block in blocks] == [2, 2, 1]
    assert [block["full_block"] for block in blocks] == [True, True, False]
    assert blocks[0]["start_token"] == 0
    assert blocks[1]["end_token_exclusive"] == 4


def test_prefix_pair_rounds_shared_tokens_down_to_full_blocks() -> None:
    evidence = prefix_pair_evidence([1, 2, 3, 4, 5], [1, 2, 3, 9], 2)
    assert evidence == {
        "common_prefix_tokens": 3,
        "cacheable_full_block_tokens": 2,
        "shared_full_blocks": 1,
        "divergence_token_index": 3,
    }


def test_best_prior_prefix_selects_the_longest_resident_candidate() -> None:
    evidence = best_prior_prefix(
        [[1, 2, 8, 9], [1, 2, 3, 4], [7, 8]],
        [1, 2, 3, 5],
        2,
    )
    assert evidence["best_prior_index"] == 1
    assert evidence["common_prefix_tokens"] == 3
    assert evidence["cacheable_full_block_tokens"] == 2


def test_best_prior_prefix_handles_first_request() -> None:
    evidence = best_prior_prefix([], [1, 2], 16)
    assert evidence["best_prior_index"] is None
    assert evidence["cacheable_full_block_tokens"] == 0


def test_incremental_prefix_index_matches_batch_evidence() -> None:
    prior = [[1, 2, 8, 9], [1, 2, 3, 4], [7, 8]]
    index = RenderedPrefixIndex(block_size=2)
    assert index.query([1, 2])["best_prior_index"] is None
    for prompt in prior:
        index.observe(prompt)
    expected = best_prior_prefix(prior, [1, 2, 3, 5], 2)
    assert index.query([1, 2, 3, 5]) == expected
