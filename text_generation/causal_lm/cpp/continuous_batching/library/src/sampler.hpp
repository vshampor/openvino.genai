
// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <cstdlib>
#include <limits>
#include <map>
#include <algorithm>
#include <cmath>
#include <random>

#include "openvino/runtime/tensor.hpp"

#include "scheduler.hpp"
#include "sequence_group.hpp"

// Modifyed Knuth–Morris–Pratt algorithm which returns tokens following after every needle occurance in haystack
std::vector<int64_t> kmp_search(const std::vector<int64_t>& haystack, const std::vector<int64_t>& needle) {
    if (needle.empty()) {  // no_repeat_ngram_size == 1, ban every token
        return {haystack.begin(), haystack.end()};
    }
    std::vector<int> partial_match_table(needle.size() + 1, -1);
    int cnd = 0;
    for (size_t pos = 1; pos < needle.size(); ++pos) {
        if (needle.at(pos) == needle.at(size_t(cnd))) {
            partial_match_table.at(pos) = partial_match_table.at(size_t(cnd));
        } else {
            partial_match_table.at(pos) = cnd;
            while (cnd >= 0 && needle.at(pos) != needle.at(size_t(cnd))) {
                cnd = partial_match_table.at(size_t(cnd));
            }
        }
        ++cnd;
    }
    partial_match_table.back() = cnd;
    std::vector<int64_t> res;
    size_t haystack_id = 0;
    int needle_id = 0;
    while (haystack_id < haystack.size() - 1) {
        if (needle.at(size_t(needle_id)) == haystack.at(haystack_id)) {
            ++haystack_id;
            ++needle_id;
            if (needle_id == int(needle.size())) {
                res.push_back(haystack.at(haystack_id));
                needle_id = partial_match_table.at(size_t(needle_id));
            }
        } else {
            needle_id = partial_match_table.at(size_t(needle_id));
            if (needle_id < 0) {
                ++haystack_id;
                ++needle_id;
            }
        }
    }
    return res;
}

struct Token {
    float m_log_prob;
    int64_t m_index;
};

std::vector<Token> log_softmax(const ov::Tensor& logits, size_t batch_idx) {
    ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3);
    size_t batch = shape[0], seq_len = shape[1], vocab_size = shape[2];
    if (!(batch_idx < batch)) {
        int a = 0;
    }
    OPENVINO_ASSERT(batch_idx < batch, "Logits batch size doesn't match the number of beams");

    size_t batch_offset = batch_idx * seq_len * vocab_size, sequence_offset = (seq_len - 1) * vocab_size;
    const float* beam_logits = logits.data<const float>() + batch_offset + sequence_offset;
    float max_logit = *std::max_element(beam_logits, beam_logits + vocab_size);
    float log_sum = std::log(std::accumulate(
        beam_logits, beam_logits + vocab_size, 0.0f, [max_logit](float accumulated, float to_add) {
            return accumulated + std::exp(to_add - max_logit);
    }));

    std::vector<Token> tokens;
    tokens.reserve(vocab_size);
    for (size_t idx = 0; idx < vocab_size; ++idx)
        tokens.push_back({beam_logits[idx] - max_logit - log_sum, int64_t(idx)});

    return tokens;
}

struct Beam {
    Sequence::Ptr m_sequence;
    size_t m_global_beam_idx = 0;

    // beam is made on top of sequence
    float m_log_prob = 0.0f;
    int64_t m_token_id = -1;

    // cumulative log probabilities
    float m_score = -std::numeric_limits<float>::infinity();

    Beam(Sequence::Ptr sequence)
        : m_sequence(sequence) { }

    size_t get_generated_len() const {
        return m_sequence->get_generated_len();
    }
};

bool greater(const Beam& left, const Beam& right) {
    return left.m_score > right.m_score;
}

struct Group {
    std::vector<Beam> ongoing;  // Best beams in front
    std::vector<Beam> min_heap;  // The worst of the best completed beams is the first
    bool done = false;

    int64_t finish(Beam beam, const GenerationConfig& sampling_params) {
        int64_t preeempted_sequence_id = -1;
        float generated_len = beam.get_generated_len() + (beam.m_token_id == sampling_params.eos_token_id ? 1 : 0); // HF counts EOS token in generation length
        beam.m_score /= std::pow(generated_len, sampling_params.length_penalty);

        min_heap.push_back(beam);
        std::push_heap(min_heap.begin(), min_heap.end(), greater);
        if (min_heap.size() > sampling_params.group_size) {
            std::pop_heap(min_heap.begin(), min_heap.end(), greater);
            preeempted_sequence_id = min_heap.back().m_sequence->get_id();
            min_heap.pop_back();
        }

        return preeempted_sequence_id;
    }

    void is_done(const GenerationConfig& sampling_params) {
        if (min_heap.size() < sampling_params.group_size)
            return;

        const Beam& best_running_sequence = ongoing.front(), & worst_finished_sequence = min_heap.front();
        size_t cur_len = best_running_sequence.m_sequence->get_generated_len();
        float best_sum_logprobs = best_running_sequence.m_score;
        float worst_score = worst_finished_sequence.m_score;
        switch (sampling_params.stop_criteria) {
        case StopCriteria::EARLY:
            done = true;
            return;
        case StopCriteria::HEURISTIC: {
            float highest_attainable_score = best_sum_logprobs / std::pow(float(cur_len), sampling_params.length_penalty);
            done = worst_score >= highest_attainable_score;
            return;
        }
        case StopCriteria::NEVER: {
            size_t length = sampling_params.length_penalty > 0.0 ? sampling_params.max_new_tokens : cur_len;
            float highest_attainable_score = best_sum_logprobs / std::pow(float(length), sampling_params.length_penalty);
            done = worst_score >= highest_attainable_score;
            return;
        }
        default:
            OPENVINO_THROW("Beam search internal error: unkown mode");
        }
    }
};

struct SamplerOutput {
    // IDs of sequences that need to be dropped (used during Beam Search)
    std::vector<uint64_t> m_dropped_sequences;
    // IDs of sequences that need to be forked (note, the same sequence can be forked multiple times)
    // it will later be used by scheduler to fork block_tables for child sequences
    // (used during Beam Search)
    std::unordered_map<uint64_t, std::list<uint64_t>> m_forked_sequences;
};

class GroupBeamSearcher {
    SequenceGroup::Ptr m_sequence_group;
    GenerationConfig m_parameters;
    std::vector<Group> m_groups;
public:
    explicit GroupBeamSearcher(SequenceGroup::Ptr sequence_group);

    void select_next_tokens(const ov::Tensor& logits, SamplerOutput& sampler_output);

    void finalize(SamplerOutput& sampler_output) {
        for (Group& group : m_groups) {
            if (!group.done) {
                for (Beam& beam : group.ongoing) {
                    uint64_t sequence_id = beam.m_sequence->get_id();

                    int64_t preempted_id = group.finish(beam, m_parameters);
                    if (preempted_id >= 0) {
                        // remove preempted one
                        m_sequence_group->remove_sequence(preempted_id);
                    }

                    // mark current sequence as finished
                    beam.m_sequence->set_status(SequenceStatus::FINISHED);
                    // we also need to drop add ongoing / forked sequences from scheduler
                    sampler_output.m_dropped_sequences.push_back(sequence_id);
                }
            }
        }
    }
};

class Sampler {

    int64_t _greedy_sample(ov::Tensor logits) const {
        ov::Shape logits_shape = logits.get_shape();
        size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2];
        OPENVINO_ASSERT(batch_size == 1);

        const float * logits_data = logits.data<const float>() + (seq_len - 1) * vocab_size;
        int64_t out_token = std::max_element(logits_data, logits_data + vocab_size) - logits_data;
        return out_token;
    }

    int64_t _multinomial_sample(ov::Tensor logits, float temperature, float top_p) {
        ov::Shape logits_shape = logits.get_shape();
        size_t batch_size = logits_shape[0], seq_len = logits_shape[1], vocab_size = logits_shape[2];
        OPENVINO_ASSERT(batch_size == 1);

        const float * logits_data = logits.data<const float>() + (seq_len - 1) * vocab_size;
        using LogitWithIdx = std::pair<float, size_t>;
        std::vector<LogitWithIdx> softmax_vector(vocab_size);
        for (size_t i = 0; i < softmax_vector.size(); i++) {
            softmax_vector[i] = LogitWithIdx(logits_data[i], i);
        }
        std::sort(softmax_vector.begin(), softmax_vector.end(), [](const LogitWithIdx& lhs, const LogitWithIdx& rhs) {return lhs.first > rhs.first; });
        float max_logit = softmax_vector[0].first;
        std::for_each(softmax_vector.begin(), softmax_vector.end(), [max_logit, temperature](LogitWithIdx& val) {val.first = expf((val.first - max_logit) / temperature);});

        float norm_sum = 0.0;
        for (const auto& val : softmax_vector) {
            norm_sum += val.first;
        }

        float probability_sum = 0.0f;
        size_t nucleus_size = 0;
        for (const auto& probability_unnormalized : softmax_vector) {
            probability_sum += probability_unnormalized.first / norm_sum;
            nucleus_size += 1;
            if (probability_sum > top_p) break;
        }

        std::vector<float> multinomial_weights(nucleus_size);

        // additionally renormalize the nucleus scores so that they add up to 1 to serve as multinomial distribution weights
        for (size_t i = 0; i < nucleus_size; i++) multinomial_weights[i] = softmax_vector[i].first / norm_sum / probability_sum;

        std::cout << "VSHAMPOR: nucleus size is " << nucleus_size << std::endl;
        std::cout << "VSHAMPOR: nucleus first value is " << multinomial_weights[0] << std::endl;
        // std::cout << "VSHAMPOR: nucleus logit values are ";
        // for (auto v : multinomial_weights) std::cout << v << " ";
        // std::cout << std::endl;

        set_seed(0);
        auto dist = std::discrete_distribution<size_t>(multinomial_weights.begin(), multinomial_weights.end()); // equivalent to multinomial with number of trials == 1
        size_t element_to_pick = dist(rng_engine);
        std::cout << "VSHAMPOR: RNG picked " << element_to_pick << std::endl;

        int64_t out_token = softmax_vector[element_to_pick].second;
        std::cout << "VSHAMPOR: out_token is " << out_token << std::endl;

        return out_token;
    }

    // request ID => beam search tracking information
    std::map<uint64_t, GroupBeamSearcher> m_beam_search_info;

    std::mt19937 rng_engine;

public:
    SamplerOutput sample(std::vector<SequenceGroup::Ptr> & sequence_groups, ov::Tensor logits);

    void set_seed(size_t seed) { rng_engine.seed(seed); }
};

SamplerOutput Sampler::sample(std::vector<SequenceGroup::Ptr> & sequence_groups, ov::Tensor logits) {
    const float * logits_data = logits.data<float>();
    ov::Shape logits_shape = logits.get_shape();
    OPENVINO_ASSERT(logits_shape.size() == 3);
    size_t batch_seq_len = logits_shape[1], vocab_size = logits_shape[2];

    SamplerOutput sampler_output;

    for (size_t sequence_group_id = 0, currently_processed_tokens = 0; sequence_group_id < sequence_groups.size(); ++sequence_group_id) {
        SequenceGroup::Ptr sequence_group = sequence_groups[sequence_group_id];
        if (!sequence_group->is_scheduled())
            continue;

        size_t num_running_sequences = sequence_group->num_running_seqs();
        size_t actual_seq_len = sequence_group->get_num_scheduled_tokens(); // points to a token which needs to be sampled
        size_t padded_amount_of_processed_tokens = std::max(actual_seq_len, batch_seq_len);
        const GenerationConfig& sampling_params = sequence_group->get_sampling_parameters();

        const void * sequence_group_logits_data = logits_data + vocab_size * currently_processed_tokens;
        ov::Tensor sequence_group_logits(ov::element::f32, ov::Shape{num_running_sequences, actual_seq_len, vocab_size}, (void *)sequence_group_logits_data);

        if (sequence_group->requires_sampling()) {
            if (sampling_params.is_greedy_sampling() || sampling_params.is_multinomial()) {
                std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
                OPENVINO_ASSERT(running_sequences.size() == 1);

                int64_t sampled_token_id;
                if (sampling_params.is_greedy_sampling()) {
                    sampled_token_id = _greedy_sample(sequence_group_logits);
                }
                else {  // .is_multinomial()
                    sampled_token_id = _multinomial_sample(sequence_group_logits, sampling_params.temperature, sampling_params.top_p);
                }
                // in case of greedy search we always have a single parent sequence to sample from
                running_sequences[0]->append_token(sampled_token_id, sequence_group_logits.data<const float>()[sampled_token_id]);

                if (sampling_params.max_new_tokens == running_sequences[0]->get_generated_len() ||
                    sampled_token_id == sampling_params.eos_token_id && !sampling_params.ignore_eos) {
                    // stop sequence by max_new_tokens or EOS token
                    running_sequences[0]->set_status(SequenceStatus::FINISHED);
                    // drop sequence from scheduler
                    sampler_output.m_dropped_sequences.push_back(running_sequences[0]->get_id());
                }
            } else if (sampling_params.is_beam_search()) {
                uint64_t request_id = sequence_group->get_request_id();

                // create beam search info if we are on the first generate
                if (m_beam_search_info.find(request_id) == m_beam_search_info.end()) {
                    m_beam_search_info.emplace(request_id, GroupBeamSearcher(sequence_group));
                }
                else {
                    // sequence group can be empty if returned after preemption
                    if (sequence_group->is_empty()) {
                        // clear beam search info
                        m_beam_search_info.erase(request_id);
                        m_beam_search_info.emplace(request_id, GroupBeamSearcher(sequence_group));
                    }
                }


                // current algorithm already adds new tokens to running sequences and
                m_beam_search_info.at(request_id).select_next_tokens(sequence_group_logits, sampler_output);

                // check max length stop criteria
                std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
                if (!sequence_group->has_finished() && running_sequences[0]->get_generated_len() == sampling_params.max_new_tokens) {
                    // stop sequence by max_new_tokens
                    m_beam_search_info.at(request_id).finalize(sampler_output);
                }
            }
        } else {
            // we are in prompt processing phase when prompt is split into chunks and processed step by step
        }

        // NOTE: it should be before 'get_num_scheduled_tokens' is used
        // update internal state of sequence group to reset scheduler tokens and update currently processed ones
        sequence_group->finish_iteration();

        // accumulate a number of processed tokens
        currently_processed_tokens += padded_amount_of_processed_tokens * num_running_sequences;
    }

    return sampler_output;
}

GroupBeamSearcher::GroupBeamSearcher(SequenceGroup::Ptr sequence_group)
    : m_sequence_group(sequence_group),
        m_parameters{m_sequence_group->get_sampling_parameters()},
        m_groups{m_parameters.num_groups} {
    OPENVINO_ASSERT(m_parameters.no_repeat_ngram_size > 0, "no_repeat_ngram_size must be positive");
    OPENVINO_ASSERT(m_sequence_group->num_running_seqs() == 1);

    for (Group& group : m_groups) {
        group.ongoing.reserve(m_parameters.group_size);
        // initially we just add our "base" sequence to beams inside each group
        for (size_t i = 0; i < m_parameters.group_size; ++i)
            group.ongoing.push_back(Beam((*sequence_group)[0]));
        // to avoid selecting the same tokens for beams within group, let's just initialize score
        // for the front one
        group.ongoing.front().m_score = 0.0f;
    }
}

void GroupBeamSearcher::select_next_tokens(const ov::Tensor& logits, SamplerOutput& sampler_output) {
    std::vector<int64_t> next_tokens;
    std::vector<int32_t> next_beams;
    next_tokens.reserve(m_parameters.num_groups * m_parameters.group_size);
    next_beams.reserve(m_parameters.num_groups * m_parameters.group_size);

    // parent sequence ID -> number of child sequences
    std::map<uint64_t, uint64_t> parent_2_num_childs_map;

    for (Group& group : m_groups) {
        if (!group.done) {
            for (Beam& beam : group.ongoing) {
                uint64_t parent_seq_id = beam.m_sequence->get_id();

                // here we need to map index of sequence in beam search group(s) and sequence group
                beam.m_global_beam_idx = [this] (uint64_t seq_id) -> size_t {
                    std::vector<Sequence::Ptr> running_seqs = m_sequence_group->get_running_sequences();
                    for (size_t seq_global_index = 0; seq_global_index < running_seqs.size(); ++seq_global_index) {
                        if (seq_id == running_seqs[seq_global_index]->get_id())
                            return seq_global_index;
                    }
                    OPENVINO_THROW("Internal error in beam search: should not be here");
                } (parent_seq_id);

                // zero out all parent forks counts
                parent_2_num_childs_map[parent_seq_id] = 0;
            }
        }
    }

    auto try_to_finish_candidate = [&] (Group& group, Beam& candidate) -> void {
        uint64_t seq_id = candidate.m_sequence->get_id();
        // try to finish candidate
        int64_t preempted_seq_id = group.finish(candidate, m_parameters);

        // if candidate has lower score than others finished
        if (preempted_seq_id == seq_id) {
            // do nothing and just ignore current finished candidate
        } else {
            if (preempted_seq_id >= 0) {
                m_sequence_group->remove_sequence(preempted_seq_id);
            }

            // need to insert candidate to a sequence group
            Sequence::Ptr forked_sequence = m_sequence_group->fork_sequence(candidate.m_sequence);
            // and finish immidiately
            forked_sequence->set_status(SequenceStatus::FINISHED);

            // TODO: make it more simplier
            // currently, we finish sequence and then fork it in current code
            {
                for (size_t i = 0; i < group.min_heap.size(); ++i) {
                    if (group.min_heap[i].m_sequence->get_id() == seq_id) {
                        group.min_heap[i].m_sequence = forked_sequence;
                        break;
                    }
                }
            }

            // HF implementation counts eos_token_id for length penalty calculation
            if (candidate.m_token_id != m_parameters.eos_token_id) {
                // append token from candidate to actual sequence
                forked_sequence->append_token(candidate.m_token_id, candidate.m_log_prob);
            }
        }
    };

    // group ID => child beams
    std::map<int, std::vector<Beam>> child_beams_per_group;

    for (size_t group_id = 0; group_id < m_groups.size(); ++group_id) {
        Group & group = m_groups[group_id];
        if (group.done)
            continue;

        std::vector<Beam> candidates;
        candidates.reserve(m_parameters.group_size * 2 * m_parameters.group_size);

        for (const Beam& beam : group.ongoing) {
            std::vector<Token> tokens = log_softmax(logits, beam.m_global_beam_idx);

            // apply diversity penalty
            for (auto prev_group_id = 0; prev_group_id < group_id; ++prev_group_id) {
                for (const Beam& prev_beam : child_beams_per_group[prev_group_id]) {
                    tokens[prev_beam.m_token_id].m_log_prob -= m_parameters.diversity_penalty;
                }
            }

            // apply n_gramm
            std::vector<int64_t> full_text{m_sequence_group->get_prompt_ids()};
            full_text.insert(full_text.end(), beam.m_sequence->get_generated_ids().begin(), beam.m_sequence->get_generated_ids().end());
            if (full_text.size() > 1 && full_text.size() >= m_parameters.no_repeat_ngram_size) {
                auto tail_start = full_text.end() - ptrdiff_t(m_parameters.no_repeat_ngram_size) + 1;
                for (int64_t banned_token : kmp_search(full_text, {tail_start, full_text.end()})) {
                    tokens[banned_token].m_log_prob = -std::numeric_limits<float>::infinity();
                }
            }

            // sort tokens
            std::sort(tokens.begin(), tokens.end(), [](Token left, Token right) {
                return left.m_log_prob > right.m_log_prob;  // Most probable tokens in front
            });

            size_t add_count = 0;
            for (Token token : tokens) {
                Beam new_candidate = beam;
                new_candidate.m_score += new_candidate.m_log_prob = token.m_log_prob;
                new_candidate.m_token_id = token.m_index;

                // TODO: fix it
                // and ensure cumulative_log prob is used
                if (/* m_parameters.early_finish(new_candidate) */ false) {
                    try_to_finish_candidate(group, new_candidate);
                } else {
                    candidates.push_back(new_candidate);
                    if (++add_count == 2 * m_parameters.group_size) {
                        break;
                    }
                }
            }
        }

        // Sample 2 * group_size highest score tokens to get at least 1 non EOS token per beam
        OPENVINO_ASSERT(candidates.size() >= 2 * m_parameters.group_size, "No beams left to search");

        auto to_sort = candidates.begin() + ptrdiff_t(2 * m_parameters.group_size);
        std::partial_sort(candidates.begin(), to_sort, candidates.end(), greater);

        for (size_t cand_idx = 0; cand_idx < candidates.size(); ++cand_idx) {
            Beam & candidate = candidates[cand_idx];
            if (m_parameters.eos_token_id == candidate.m_token_id) {
                // If beam_token does not belong to top num_beams tokens, it should not be added
                if (cand_idx >= m_parameters.group_size)
                    continue;

                // try to finish candidate
                try_to_finish_candidate(group, candidate);
            } else {
                parent_2_num_childs_map[candidate.m_sequence->get_id()] += 1;
                child_beams_per_group[group_id].push_back(candidate);

                // if num childs are enough
                if (child_beams_per_group[group_id].size() == m_parameters.group_size) {
                    break;
                }
            }
        }

        // check whether group has finished
        group.is_done(m_parameters);
        if (group.done) {
            // group has finished, group all running sequences
            for (const Beam& beam : group.ongoing) {
                uint64_t seq_id = beam.m_sequence->get_id();
                m_sequence_group->remove_sequence(seq_id);
                sampler_output.m_dropped_sequences.push_back(seq_id);
            }
            group.ongoing.clear();
        }
    }

    // fork child sequences for non-finished groups

    for (size_t group_id = 0; group_id < m_groups.size(); ++group_id) {
        Group & group = m_groups[group_id];

        if (!group.done) {
            for (Beam& child_beam : child_beams_per_group[group_id]) {
                uint64_t parent_sequence_id = child_beam.m_sequence->get_id();
                size_t& num_childs = parent_2_num_childs_map[parent_sequence_id];

                // if current beam is forked multiple times
                if (num_childs > 1) {
                    child_beam.m_sequence = m_sequence_group->fork_sequence(child_beam.m_sequence);
                    child_beam.m_sequence->append_token(child_beam.m_token_id, child_beam.m_log_prob);

                    // reduce forks count, since fork already happened and next loop iteration
                    // will go by the second branch (num_childs == 1)
                    --num_childs;

                    // fill out sampler output
                    sampler_output.m_forked_sequences[parent_sequence_id].push_back(child_beam.m_sequence->get_id());
                } else if (num_childs == 1) {
                    // keep current sequence going and add a new token
                    child_beam.m_sequence->append_token(child_beam.m_token_id, child_beam.m_log_prob);
                }
            }

            // drop beams which are not forked by current group
            for (const Beam& beam : group.ongoing) {
                size_t num_childs = parent_2_num_childs_map[beam.m_sequence->get_id()];
                if (num_childs == 0) {
                    // drop sequence as not forked
                    sampler_output.m_dropped_sequences.push_back(beam.m_sequence->get_id());
                    m_sequence_group->remove_sequence(beam.m_sequence->get_id());
                }
            }

            // child become parents
            group.ongoing = child_beams_per_group[group_id];
        }
    }
}
