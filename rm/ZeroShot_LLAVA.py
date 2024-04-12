'''
Zero shot LLAVA

pip install -q bitsandbytes accelerate
pip install gym-super-mario-bros
pip install imageio
pip install imageio-ffmpeg
'''
from transformers import AutoTokenizer, AutoProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import requests
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import imageio
import numpy as np


env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

with open("/content/Supermario_prompt_1.txt", "r") as f:
  prompt = f.read()

tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-1.5-7b-hf")
model = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    load_in_4bit=True
)

obs = env.reset()
images = []
ret = 0
action = -1
for i in range(100):
    # action = env.action_space.sample()
    inputs = processor(prompt, obs, padding=True, truncation=True, return_tensors='pt', max_length=1024).to(0, torch.float16)
    output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
    out_action = processor.decode(output[0], skip_special_tokens=True).split('ASSISTANT:')[-1]
    for j in range(7):
      if str(j) in out_action:
        action = j
        print(action)
    for _ in range(4): # hyper parameter
      obs, reward, done, info = env.step(action)
    # obs
      images.append(obs.copy())
      ret += reward
      print('-----{}----'.format(i))
      if done:
          print('here')
          obs = env.reset()

# Save the video
imageio.mimsave("mario_video.mp4", np.array(images), fps=30)