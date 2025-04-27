import torch
import requests
from PIL import Image
from modelscope import MllamaForConditionalGeneration, AutoProcessor

# 函数1：加载模型
def load_model(model_id="LLM-Research/Llama-3.2-11B-Vision-Instruct"):
    """
    加载 Llama-3.2-Vision-Instruct 模型和处理器。
    :param model_id: 模型名称或本地路径
    :return: 加载好的模型和处理器
    """
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

# 函数2：模型推理
def get_model_output(prompt, image_path_or_url, model, processor):
    """
    使用加载好的模型和处理器进行推理。
    :param prompt: 提示文本
    :param image_path_or_url: 图片路径或图片URL
    :param model: 加载好的模型
    :param processor: 加载好的处理器
    :return: 模型生成的输出文本
    """
    # 处理图片
    if image_path_or_url.startswith("http"):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")

    # 构建输入消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 准备模型输入
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    # 生成输出
    output = model.generate(**inputs, max_new_tokens=128)
    output_text = processor.decode(output[0], skip_special_tokens=True)

    return output_text
