import os
import sys
import random
import base64
import requests
from pathlib import Path

# 添加项目根目录到运行路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.append(str(project_root))

from config import DATA_BASELINE_DIR, DATASET_ROOT

# API 配置 
API_KEY = "sk-bpiqdrinpjbwiqcbdejehuqlhyiwgluksivuiisiolkympra"
API_URL = "https://api.siliconflow.cn/v1/images/generations"

# 存放 AIGC 生成图片的新目录
AIGC_OUTPUT_DIR = DATASET_ROOT / "aigc_image" / "images"
os.makedirs(AIGC_OUTPUT_DIR, exist_ok=True)

# 随机 Prompt 模板库 
BASE_PROMPT = "Eye-level shot, horizontal view of the vast ocean, maritime photography, photorealistic, 8k resolution"

# 负面提示词 
NEGATIVE_PROMPT = "top-down view, aerial photography, satellite image, bird's-eye view, huge ships, giant cruise, text, watermark, cartoon, 3d render, painting"

# 模板库：天气与光照
WEATHER_LIGHTING_TEMPLATES = [
    "clear sky, bright daylight, bright sun glare on water",
    "heavy sea fog, misty ocean, bleak weather, very low visibility",
    "sunset, golden hour, specular highlights on the water surface, backlit silhouettes",
    "overcast sky, gloomy lighting, dark clouds, impending storm",
    "raining heavily, dark stormy sea, cinematic lighting"
]

# 模板库：海浪状态
SEA_STATE_TEMPLATES = [
    "calm sea, gentle ripples, highly detailed water surface",
    "rough ocean waves, strong wind, prominent whitecaps and sea spray",
    "turbulent sea water, dynamic water texture",
    "choppy dark green seawater, subtle waves"
]

# 模板库：小目标描述
TARGET_DESC_TEMPLATES = [
    "a few tiny distant ships on the far horizon",
    "faint blurry silhouettes of small distant boats barely visible",
    "tiny fishing boats struggling far away",
    "distant small vessels scattered near the horizon line"
]

#生成prompt
def generate_random_prompt():
    weather = random.choice(WEATHER_LIGHTING_TEMPLATES)
    sea_state = random.choice(SEA_STATE_TEMPLATES)
    target = random.choice(TARGET_DESC_TEMPLATES)
    final_prompt = f"{BASE_PROMPT}, {weather}, {sea_state}, {target}."
    return final_prompt


# 图像转Base64编码
def image_to_base64(img_path):
    """读取图片并转换为 base64 编码"""
    with open(img_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

#调用api
def generate_img2img(img_path, output_path, denoising_strength=0.55):
    prompt = generate_random_prompt()
    print(f"当前Prompt: {prompt}")
    
    # 提取并格式化base64
    init_image_b64 = image_to_base64(img_path)
    image_data = f"data:image/jpeg;base64,{init_image_b64}"
    
    # 构建请求头部
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # 构建请求payload，基于所提供的手册
    payload = {
        "model": "Qwen/Qwen-Image-Edit-2509", # 使用 Qwen 的图生图模型
        "prompt": prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "image": image_data,
        "num_inference_steps": 25,
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        
        r = response.json()
        img_info = r['images'][0]
        
        if 'url' in img_info:
            img_url = img_info['url']
            img_res = requests.get(img_url)
            img_res.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(img_res.content)
        elif 'b64_json' in img_info or 'base64' in img_info:
            b64_data = img_info.get('b64_json', img_info.get('base64'))
            img_data = base64.b64decode(b64_data)
            with open(output_path, "wb") as f:
                f.write(img_data)
                
        print(f"[+] 成功生成图片并保存至: {output_path}")
        
    except Exception as e:
        print(f"[-] API 请求失败: {e}")
        if 'response' in locals() and hasattr(response, 'text'):
            print(f"    服务器返回: {response.text}")

# 批量处理目录下的图片
def main():
    # 从基线数据集中的训练集中随机挑选一些图作为底图
    input_dir = DATA_BASELINE_DIR / "images" / "train"
    
    if not input_dir.exists():
        print(f"找不到输入目录 {input_dir}，请检查是否已生成 baseline 数据")
        return
        
    # 获取所有的 .jpg 图
    image_list = list(input_dir.glob("*.jpg"))
    if not image_list:
        print("输入目录中没有图片")
        return
        
    # 随机抽取100张图
    sample_images = random.sample(image_list, min(100, len(image_list)))
    
    print(f"开始进行图生图，共挑选 {len(sample_images)} 张底图...")
    for idx, img_path in enumerate(sample_images):
        print(f"\n--- 处理进度 {idx+1}/{len(sample_images)} ---")
        output_filename = f"aigc_aug_{img_path.name}"
        output_path = AIGC_OUTPUT_DIR / output_filename
        
        # 开始生成
        generate_img2img(img_path, output_path, denoising_strength=0.55)

if __name__ == "__main__":
    main()
