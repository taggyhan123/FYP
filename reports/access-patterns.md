# Tool-access, replay, and trie analysis

This report keeps ToolRet gold relevance and BFCL menu exposure in separate partitions. Ordered adjacency is only an order proxy; neither dataset provides a natural production execution trace.

## toolret_gold

| Tasks | Unique observed tools | Tools/task median | Tools/task P95 | Schema tokens/task median | Schema tokens/task P95 |
| --- | --- | --- | --- | --- | --- |
| 7961 | 7652 | 1.0 | 4.0 | 124.0 | 387.0 |

### Highest benchmark support

| Tool | Occurrences | Schema tokens | Weighted tokens |
| --- | --- | --- | --- |
| Finish | 350 | 33 | 11550 |
| finish | 170 | 49 | 8330 |
| get_closing_parenthesis | 70 | 46 | 3220 |
| stack_insert | 70 | 49 | 3430 |
| stack_pop | 70 | 44 | 3080 |
| account_login | 64 | 76 | 4864 |
| API.select_booking_type | 63 | 67 | 4221 |
| create_object_dict | 62 | 53 | 3286 |
| update_object_dict | 62 | 53 | 3286 |
| get_final_object | 62 | 47 | 2914 |

### Co-occurrence and ordered-adjacency proxies

Co-occurrence is an unordered same-task statistic. Adjacency uses dataset label/menu order and is not a measured execution transition.

| Top pair | Tasks |
| --- | --- |
| get_closing_parenthesis + stack_insert | 70 |
| get_closing_parenthesis + stack_pop | 70 |
| stack_insert + stack_pop | 70 |
| create_object_dict + update_object_dict | 62 |
| create_object_dict + get_final_object | 62 |
| update_object_dict + get_final_object | 62 |
| divide_remain + check_validity | 38 |
| update_orientation + update_location | 30 |
| multiply + add_subtract_hadamard | 23 |
| clock_alarm_cancel + clock_alarm_set | 22 |

| Top triple | Tasks |
| --- | --- |
| get_closing_parenthesis + stack_insert + stack_pop | 70 |
| create_object_dict + update_object_dict + get_final_object | 62 |
| multiply + add_subtract_hadamard + kronecker | 13 |
| multiply + add_subtract_hadamard + transpose | 12 |
| sum_over_axis + multiply + add_subtract_hadamard | 10 |
| sum_over_axis + add_subtract_hadamard + kronecker | 10 |
| URL_Link_Shortener_Create_a_new_link + URL_Link_Shortener_Get_a_list_of_domains + bitly_shorten | 8 |
| PPT.create_file + PPT.save_file + PPT.add_text_page | 7 |
| TextSummarizer + EntityFrequencyCounter + Finish | 7 |
| sum_over_axis + multiply + kronecker | 7 |

| Ordered adjacency | Count | P(next \| current) |
| --- | --- | --- |
| get_closing_parenthesis → stack_insert | 70 | 100.00% |
| stack_insert → stack_pop | 70 | 100.00% |
| create_object_dict → update_object_dict | 62 | 100.00% |
| update_object_dict → get_final_object | 62 | 100.00% |
| check_validity → divide_remain | 38 | 100.00% |
| update_orientation → update_location | 30 | 100.00% |
| search_train → train_ticket_booking | 20 | 100.00% |
| add_subtract_hadamard → multiply | 18 | 51.43% |
| file_write → file_modify | 18 | 72.00% |
| bitly_shorten → URL_Link_Shortener_Get_a_list_of_domains | 17 | 100.00% |

### Empirical-order trie comparison

| Ordering | Trie nodes | Node compression | Block-cacheable tokens | Estimated token reuse |
| --- | --- | --- | --- | --- |
| original | 9952 | 29.45% | 353312 | 26.91% |
| alphabetical | 9766 | 30.77% | 352688 | 26.86% |
| random_seed_7 | 9911 | 29.74% | 340976 | 25.97% |
| random_seed_42 | 9753 | 30.86% | 350592 | 26.70% |
| random_seed_101 | 9909 | 29.75% | 347024 | 26.43% |
| frequency | 8904 | 36.88% | 411584 | 31.35% |
| schema_cost_weighted | 8954 | 36.52% | 417488 | 31.80% |
| fp_tree_global | 8904 | 36.88% | 411584 | 31.35% |

FP-tree-style global ordering is descending global support, so it is intentionally identical to the frequency baseline in this first implementation. Future conditional FP-tree mining should be evaluated as a distinct extension.

### Replay locality

| Replay | Same-domain adjacency | Shared-tool adjacency | Mean tool Jaccard |
| --- | --- | --- | --- |
| empirical | 99.86% | 8.94% | 0.0358 |
| uniform | 48.71% | 0.34% | 0.0013 |
| skewed | 39.35% | 24.16% | 0.0535 |
| session_bursty | 99.97% | 1.93% | 0.0058 |

### Best ordering per replay (analytical estimate)

| Replay | Ordering | Estimated token reuse | Node compression |
| --- | --- | --- | --- |
| empirical | schema_cost_weighted | 31.80% | 36.52% |
| uniform | schema_cost_weighted | 52.12% | 56.65% |
| skewed | schema_cost_weighted | 80.18% | 88.88% |
| session_bursty | schema_cost_weighted | 31.80% | 36.52% |

Pair/triple calculations skipped 0 tasks with more than 25 exposed tools to avoid combinatorial distortion.

## bfcl_exposed

| Tasks | Unique observed tools | Tools/task median | Tools/task P95 | Schema tokens/task median | Schema tokens/task P95 |
| --- | --- | --- | --- | --- | --- |
| 1240 | 1362 | 1.0 | 3.0 | 119.0 | 362.2 |

### Highest benchmark support

| Tool | Occurrences | Schema tokens | Weighted tokens |
| --- | --- | --- | --- |
| weather_forecast | 10 | 77 | 770 |
| event_finder.find_upcoming | 8 | 111 | 888 |
| get_stock_info | 7 | 109 | 763 |
| chess.rating | 7 | 97 | 679 |
| get_scientist_for_discovery | 7 | 65 | 455 |
| mutation_type.find | 7 | 108 | 756 |
| soccer.get_last_match | 7 | 89 | 623 |
| math.gcd | 6 | 66 | 396 |
| musical_scale | 6 | 79 | 474 |
| mix_paint_color | 6 | 114 | 684 |

### Co-occurrence and ordered-adjacency proxies

Co-occurrence is an unordered same-task statistic. Adjacency uses dataset label/menu order and is not a measured execution transition.

| Top pair | Tasks |
| --- | --- |
| math.circle_area + math.triangle_area_base_height | 2 |
| country_info.capital + country_info.largest_city | 2 |
| country_info.capital + country_info.population | 2 |
| country_info.largest_city + country_info.population | 2 |
| weather.get_by_city_date + weather.get_by_coordinates_date | 2 |
| weather.get_by_city_date + weather.get_forecast_by_coordinates | 2 |
| weather.get_by_coordinates_date + weather.get_forecast_by_coordinates | 2 |
| ecological_impact.analyze + wildlife_population.assess_growth | 2 |
| property_valuation.get + realestate.find_properties | 2 |
| calculate_average + calculate_standard_deviation | 2 |

| Top triple | Tasks |
| --- | --- |
| country_info.capital + country_info.largest_city + country_info.population | 2 |
| weather.get_by_city_date + weather.get_by_coordinates_date + weather.get_forecast_by_coordinates | 2 |
| calculate_average + calculate_standard_deviation + highest_grade | 2 |
| sculptor_info.get + sculpture_availability.check + sculpture_price.calculate | 2 |
| sports_data.basketball.most_points_career + sports_data.basketball.most_points_single_game + sports_data.basketball.most_points_single_season | 2 |
| convert.rgb_to_hex + perform.string_reverse + solve.quadratic_equation | 2 |
| geometry_circle.calculate + geometry_rectangle.calculate + geometry_square.calculate | 2 |
| avg_closing_price + total_revenue + volume_traded | 2 |
| air_quality_forecast + news + weather_forecast | 2 |
| games.price.find + games.reviews.find + games.update.find | 2 |

| Ordered adjacency | Count | P(next \| current) |
| --- | --- | --- |
| weather.get_forecast_by_coordinates → weather.get_by_coordinates_date | 2 | 100.00% |
| wildlife_population.assess_growth → ecological_impact.analyze | 2 | 100.00% |
| property_valuation.get → realestate.find_properties | 2 | 100.00% |
| sculptor_info.get → sculpture_price.calculate | 2 | 100.00% |
| geology.get_era → history.get_event_date | 2 | 100.00% |
| cosine_similarity.calculate → correlation.calculate | 2 | 100.00% |
| volume_traded → total_revenue | 2 | 100.00% |
| total_revenue → avg_closing_price | 2 | 100.00% |
| air_quality_forecast → weather_forecast | 2 | 100.00% |
| timezones.get_difference → geodistance.find | 2 | 100.00% |

### Empirical-order trie comparison

| Ordering | Trie nodes | Node compression | Block-cacheable tokens | Estimated token reuse |
| --- | --- | --- | --- | --- |
| original | 1689 | 11.89% | 22144 | 10.98% |
| alphabetical | 1598 | 16.64% | 31008 | 15.38% |
| random_seed_7 | 1600 | 16.54% | 29440 | 14.60% |
| random_seed_42 | 1606 | 16.22% | 28720 | 14.25% |
| random_seed_101 | 1607 | 16.17% | 29408 | 14.59% |
| frequency | 1513 | 21.07% | 38064 | 18.88% |
| schema_cost_weighted | 1523 | 20.55% | 37824 | 18.76% |
| fp_tree_global | 1513 | 21.07% | 38064 | 18.88% |

FP-tree-style global ordering is descending global support, so it is intentionally identical to the frequency baseline in this first implementation. Future conditional FP-tree mining should be evaluated as a distinct extension.

### Replay locality

| Replay | Same-domain adjacency | Shared-tool adjacency | Mean tool Jaccard |
| --- | --- | --- | --- |
| empirical | 99.68% | 0.65% | 0.0020 |
| uniform | 22.68% | 0.08% | 0.0008 |
| skewed | 23.24% | 0.48% | 0.0023 |
| session_bursty | 99.68% | 0.00% | 0.0000 |

### Best ordering per replay (analytical estimate)

| Replay | Ordering | Estimated token reuse | Node compression |
| --- | --- | --- | --- |
| empirical | frequency | 18.88% | 21.07% |
| uniform | frequency | 41.11% | 44.14% |
| skewed | frequency | 61.39% | 64.00% |
| session_bursty | frequency | 18.88% | 21.07% |

Pair/triple calculations skipped 0 tasks with more than 25 exposed tools to avoid combinatorial distortion.

## Interpretation boundary

The cache result is an analytical tool-unit trie estimate. It rounds shared canonical tool tokens down to 16-token blocks and assumes an unbounded retained trie. It excludes constant system/user prefixes, chat-template separators, eviction, GPU pressure, and scheduler effects. The CUDA vLLM probe in `cluster/` is required before making latency or cache-hit claims.

A useful positive signal is ordering-dependent block reuse on multi-tool workloads, especially under bursty replay. A negative or flat ToolRet result is also expected when most relevance sets contain one tool; reordering cannot improve a one-item sequence.
