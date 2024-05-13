import pytest
from pathlib import Path
from typing import List, Tuple
from unittest import mock
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig as HFGenerationConfig
from optimum.intel import OVModelForCausalLM

from py_continuous_batching import ContinuousBatchingPipeline, GenerationConfig, SchedulerConfig, GenerationResult

def get_greedy() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_return_sequences = 1
    return generation_config

def get_beam_search() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.num_groups = 3
    generation_config.group_size = 2
    generation_config.max_new_tokens = 30
    generation_config.num_return_sequences = generation_config.num_groups * generation_config.group_size
    return generation_config

def get_multinomial_temperature() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    return generation_config

def get_multinomial_temperature_and_top_p() -> GenerationConfig:
    generation_config = GenerationConfig()
    generation_config.do_sample = True
    generation_config.temperature = 0.8
    generation_config.top_p = 0.2
    return generation_config

def get_test_dataset() -> Tuple[List[str], List[GenerationConfig]]:
    prompts = [
            "What is OpenVINO?",
            "How are you?",
            "What is your name?",
            "Tell me something about Canada"
            ]
    generation_configs = [
            get_greedy(),
            get_beam_search(),
            get_greedy(),
            get_beam_search()
            ]
    return (prompts, generation_configs)

def get_scheduler_config(scheduler_params: dict = None) -> SchedulerConfig:
    scheduler_config = SchedulerConfig()
    if scheduler_params is None:
        scheduler_config.dynamic_split_fuse = True
        scheduler_config.num_kv_blocks = 300
        # vLLM specific
        scheduler_config.max_num_batched_tokens = 256
        scheduler_config.max_num_seqs = 256
    else:
        for param, value in scheduler_params.items():
            setattr(scheduler_config, param, value)

    return scheduler_config

def convert_to_hf(
        default_generation_config : HFGenerationConfig,
        generation_config : GenerationConfig
        ) -> HFGenerationConfig:
    kwargs = {}

    # generic parameters
    kwargs['max_length'] = generation_config.max_length
    kwargs['max_new_tokens'] = generation_config.max_new_tokens

    # copy default parameters
    kwargs['eos_token_id'] = default_generation_config.eos_token_id
    kwargs['pad_token_id'] = default_generation_config.pad_token_id

    if generation_config.num_groups * generation_config.group_size > 1:
        # beam search case
        kwargs['num_beam_groups'] = generation_config.num_groups
        kwargs['num_beams'] = generation_config.num_groups * generation_config.group_size
        kwargs['diversity_penalty'] = generation_config.diversity_penalty
        kwargs['repetition_penalty'] = generation_config.repetition_penalty
        kwargs['length_penalty'] = generation_config.length_penalty
        kwargs['no_repeat_ngram_size'] = generation_config.no_repeat_ngram_size
        kwargs['num_return_sequences'] = generation_config.num_return_sequences
        kwargs['output_scores'] = True
    elif generation_config.do_sample:
        # mulitinomial
        kwargs['temperature'] = generation_config.temperature
        kwargs['top_k'] = generation_config.top_k
        kwargs['top_p'] = generation_config.top_p
        kwargs['do_sample'] = generation_config.do_sample
    else:
        # greedy
        pass

    hf_generation_config = HFGenerationConfig(**kwargs)
    return hf_generation_config

def run_hugging_face(
        model_id : str,
        prompts: List[str],
        generation_configs: List[GenerationConfig],
        use_optimum: bool,
        tmp_path: Path
        ) -> Tuple[List[GenerationResult], str]:
    hf_tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = OVModelForCausalLM.from_pretrained(model_id, export=True) if use_optimum else \
            AutoModelForCausalLM.from_pretrained(model_id)
    generation_results: List[GenerationResult] = []
    model_path : Path = tmp_path / model_id

    if use_optimum:
        model.save_pretrained(model_path)
        # convert tokenizers as well
        from openvino_tokenizers import convert_tokenizer
        from openvino import serialize
        tokenizer, detokenizer = convert_tokenizer(hf_tokenizer, with_detokenizer=True)
        serialize(tokenizer, model_path / "openvino_tokenizer.xml")
        serialize(detokenizer, model_path / "openvino_detokenizer.xml")

    for prompt, generation_config in zip(prompts, generation_configs):
        inputs = hf_tokenizer(prompt, return_tensors="pt")
        prompt_len = len(inputs['input_ids'][0])
        with mock.patch('torch.multinomial', numpy_multinomial):
            generate_outputs = model.generate(**inputs, generation_config=convert_to_hf(model.generation_config, generation_config), return_dict_in_generate=True)
        all_text_batch = hf_tokenizer.batch_decode([generated_ids[prompt_len:] for generated_ids in generate_outputs.sequences], skip_special_tokens=True)

        generation_result = GenerationResult()
        generation_result.m_generation_ids = all_text_batch
        # sequences_scores are available only for beam search case
        if generation_config.is_beam_search:
            generation_result.m_scores = [score for score in generate_outputs.sequences_scores]
        generation_results.append(generation_result)

    return (generation_results, model_path)

def run_continuous_batching(
        model_path : Path,
        scheduler_config : SchedulerConfig,
        prompts: List[str],
        generation_configs : List[GenerationConfig]
        ) -> List[GenerationResult]:
    pipe = ContinuousBatchingPipeline(model_path.absolute().as_posix(), scheduler_config)
    return pipe.generate(prompts, generation_configs)


def _generate_and_compare_with_hf(model_id: str, prompts: List[str], generation_configs: List[GenerationConfig], scheduler_config: SchedulerConfig, tmp_path: Path):
    (hf_results, model_path) = run_hugging_face(model_id=model_id, prompts=prompts, generation_configs=generation_configs, tmp_path=tmp_path, use_optimum=True)
    my_results : List[GenerationResult] = run_continuous_batching(model_path, scheduler_config, prompts, generation_configs)

    assert len(prompts) == len(hf_results)
    assert len(prompts) == len(my_results)

    for prompt, hf_result, my_result, generation_config in zip(prompts, hf_results, my_results, generation_configs):
        print(f"Prompt = {prompt}\nHF result = {hf_result}\nmy result = {my_result}")

        if generation_config.is_beam_search:
            assert len(hf_result.m_scores) == len(my_result.m_scores)
            for hf_score, my_score in zip(hf_result.m_scores, my_result.m_scores):
                # Note, that for fp32 / fp16 models scores are different less than 0.001
                assert abs(hf_score - my_score) < 0.02

        assert len(hf_result.m_generation_ids) == len(my_result.m_generation_ids)
        for hf_text, my_text in zip(hf_result.m_generation_ids, my_result.m_generation_ids):
            assert hf_text == my_text

def numpy_multinomial(inputs, num_samples):
    print("VSHAMPOR: mock RNG called")
    #import pdb
    #pdb.set_trace()
    input_copy = inputs.clone().detach()
    inputs_numpy = input_copy.cpu().numpy().squeeze(0).astype(np.float64)
    inputs_numpy /= inputs_numpy.sum()  # numpy loses precision when converting and the input sum becomes >= 1
    np.random.seed(0)
    retval_numpy = np.random.multinomial(num_samples, inputs_numpy).nonzero()[0]  # numpy returns sums of one-hots, not indices like torch
    import torch
    return torch.from_numpy(retval_numpy).to(inputs.device).unsqueeze(0)


@pytest.mark.parametrize("generation_config", [get_greedy(), get_beam_search(), get_multinomial_temperature(), get_multinomial_temperature_and_top_p()],
        ids=["greedy", "beam", "multinomial_temperature", "multinomial_temperature_and_top_p"])
def test_individual_generation_configs(tmp_path, generation_config):
    prompts = [
            "What is OpenVINO?",
            ]
    generation_config.rng_seed = 0
    generation_configs = [generation_config]
    model_id : str = "facebook/opt-125m"
    _generate_and_compare_with_hf(model_id, prompts, generation_configs, DEFAULT_SCHEDULER_CONFIG, tmp_path)

# tested models:
# - facebook/opt-125m
# - meta-llama/Llama-2-7b-chat-hf
# - mistralai/Mistral-7B-Instruct-v0.2

scheduler_params_list = [{"num_kv_blocks": 300, "block_size": 16, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256},
                         {"num_kv_blocks": 40, "block_size": 4, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256}, # test preemption for dynamic_split_fuse
                         {"num_kv_blocks": 40, "block_size": 4, "dynamic_split_fuse": False, "max_num_batched_tokens": 256, "max_num_seqs": 256}] # test preemption for vllm

DEFAULT_SCHEDULER_CONFIG = get_scheduler_config({"num_kv_blocks": 300, "block_size": 16, "dynamic_split_fuse": True, "max_num_batched_tokens": 256, "max_num_seqs": 256})

@pytest.mark.parametrize("scheduler_params", scheduler_params_list)
def test_preemption(tmp_path, scheduler_params):
    prompts, generation_configs = get_test_dataset()
    model_id : str = "facebook/opt-125m"
    scheduler_config = get_scheduler_config(scheduler_params)
    _generate_and_compare_with_hf(model_id, prompts, generation_configs, scheduler_config, tmp_path)



