# vllm_model.py
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoProcessor
import os
import json
from qwen_vl_utils import process_vision_info

def load_model(model_path,  max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=20000):
    # 初始化 vLLM 推理引擎
    #llm = LLM(model=model_path, tokenizer=model_path, max_model_len=max_model_len,trust_remote_code=True)
    llm = LLM(model=model_path, gpu_memory_utilization=0.98, limit_mm_per_prompt={"image": 10, "video": 10}, max_model_len=max_model_len)
    processor = AutoProcessor.from_pretrained(model_path)
    return llm, processor
    

def get_model_output(prompts, image_path, llm, processor, max_tokens=512, temperature=0.8, top_p=0.95, max_model_len=2048):


    # 准备输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompts},
            ],
        }
    ]

     # 准备模型输入
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    # inputs = processor(
    #     text=[text],
    #     images=image_inputs,
    #     videos=video_inputs,
    # )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    llm_inputs = {
    "prompt": text,
    "multi_modal_data": mm_data,
    }
    #inputs = inputs.to("cuda")
    
    #stop_token_ids = [151329, 151336, 151338]
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    #sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop_token_ids=stop_token_ids)
    # 初始化 vLLM 推理引擎

    sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=256,
    stop_token_ids=[],
    )

    generated_ids = llm.generate([llm_inputs], sampling_params=sampling_params)

    # # 生成输出
    # #generated_ids = model.generate(**inputs, max_new_tokens=128)
    # generated_ids_trimmed = [
    #     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # output_text = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )[0]
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    output_text = outputs[0].outputs[0].text
    
    return output_text

 
