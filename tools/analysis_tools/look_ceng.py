from mmseg.apis import init_model
from mmseg.utils import register_all_modules

import torch
# 1. 配置路径（替换为你的文件）
config_path = ''
checkpoint_path = ''  # 可选，仅看结构可传None
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# 2. 初始化模型
register_all_modules()  # mmseg必须注册模块
model = init_model(config_path, checkpoint_path, device=device)
model.eval()  # 推理模式

# 3. 遍历并打印所有层（名称+模块类型）
print("=== 模型所有层的路径+类型 ===")
for idx, (name, module) in enumerate(model.named_modules()):
    # name：层的完整路径（如backbone.stem.conv1）；module：层对象（如Conv2d）
    print(f"[{idx}] 层路径：{name:50s} | 模块类型：{module.__class__.__name__}")

#backbone.block1.self_attention.conv2d
#backbone.block2.self_attention.conv2d
#backbone.lskaronghe1.convx.conv
#backbone.lskaronghe1.convy.conv
#backbone.lskaronghe1.LSKA1.conv1
#backbone.lskaronghe1.LSKA2.conv1
#backbone.lskaronghe2.convx.conv
#backbone.lskaronghe2.convy.conv
#backbone.lskaronghe2.LSKA1.conv1
#backbone.lskaronghe2.LSKA2.conv1