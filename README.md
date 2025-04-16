# elm
饿了么消防隐患识别--qwen2.5vl 7B lora微调， 微调框架--llama factory

项目各文件夹解析：  
    data_aug: 数据增强，对原始图片进行各种增强，比如翻转，色彩抖动等。  

    data_process: 视觉大模型数据集处理方法，制作用于训练的数据集格式，采用ShareGpt格式。  

    model_inf: 经过llama factory微调后的模型推理代码，这里是直接用qwen的方法加载模型进行推理。  

    vllm_model_inf: 部署vllm进行推理加速，径llama factory微调后的模型可以直接进行vllm框架推理。
