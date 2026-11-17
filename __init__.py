"""ComfyUI-MOSS-MultiID

给 MOSS-Image 做多个人物各自生效、不串脸的辅助节点。
只新增本文件夹，不修改 ComfyUI 本体，不修改任何模型权重。
删除本文件夹即可完全还原。
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 兼容补丁：transformers 5.x 把 all_tied_weights_keys 放在 PreTrainedModel.__init__ 里逐个实例赋值，
# 而不少"远程代码"模型（例如 Sa2VA 的 modeling_sa2va_qwen.py，按 4.x 写的）不会走到那一步，
# 于是报 "object has no attribute 'all_tied_weights_keys'"。
# 这里补一个类级默认值：实例上有的照旧覆盖，没有的拿这个空字典兜底。只影响找不到该属性的模型。
try:
    from transformers.modeling_utils import PreTrainedModel as _PTM
    if not hasattr(_PTM, "all_tied_weights_keys"):
        _PTM.all_tied_weights_keys = {}
except Exception:
    pass


try:
    from .nodes import NODE_CLASS_MAPPINGS as _A, NODE_DISPLAY_NAME_MAPPINGS as _B
    NODE_CLASS_MAPPINGS.update(_A)
    NODE_DISPLAY_NAME_MAPPINGS.update(_B)
except Exception as exc:
    print(f"[MOSS多身份] 基础节点加载失败，原因：{exc}")

try:
    from .gated_lora import (NODE_CLASS_MAPPINGS_EXTRA as _C,
                             NODE_DISPLAY_NAME_MAPPINGS_EXTRA as _D)
    NODE_CLASS_MAPPINGS.update(_C)
    NODE_DISPLAY_NAME_MAPPINGS.update(_D)
except Exception as exc:
    print(f"[MOSS多身份] 分区滤镜节点加载失败，原因：{exc}")

try:
    from .face_region import (NODE_CLASS_MAPPINGS as _E,
                              NODE_DISPLAY_NAME_MAPPINGS as _F)
    NODE_CLASS_MAPPINGS.update(_E)
    NODE_DISPLAY_NAME_MAPPINGS.update(_F)
except Exception as exc:
    print(f"[MOSS多身份] 人脸定位地盘节点加载失败，原因：{exc}")



try:
    from .smallface_nodes import (NODE_CLASS_MAPPINGS as _S,
                                  NODE_DISPLAY_NAME_MAPPINGS as _S2)
    NODE_CLASS_MAPPINGS.update(_S)
    NODE_DISPLAY_NAME_MAPPINGS.update(_S2)
except Exception as exc:
    print(f"[MOSS多身份] 小脸裁剪放大/贴回节点加载失败，原因：{exc}")

try:
    from .preset_check import (NODE_CLASS_MAPPINGS as _L,
                               NODE_DISPLAY_NAME_MAPPINGS as _M)
    NODE_CLASS_MAPPINGS.update(_L)
    NODE_DISPLAY_NAME_MAPPINGS.update(_M)
except Exception as exc:
    print(f"[MOSS多身份] 预设管理与出图核对节点加载失败，原因：{exc}")


try:
    from .translate_api import (NODE_CLASS_MAPPINGS as _T,
                                NODE_DISPLAY_NAME_MAPPINGS as _U)
    NODE_CLASS_MAPPINGS.update(_T)
    NODE_DISPLAY_NAME_MAPPINGS.update(_U)
except Exception as exc:
    print(f"[MOSS多身份] API版转构图节点加载失败，原因：{exc}")

try:
    from .switch_nodes import (NODE_CLASS_MAPPINGS as _W,
                               NODE_DISPLAY_NAME_MAPPINGS as _W2)
    NODE_CLASS_MAPPINGS.update(_W)
    NODE_DISPLAY_NAME_MAPPINGS.update(_W2)
except Exception as exc:
    print(f"[MOSS多身份] 开关节点加载失败，原因：{exc}")

try:
    from .person_masks import (NODE_CLASS_MAPPINGS as _P,
                               NODE_DISPLAY_NAME_MAPPINGS as _Q)
    NODE_CLASS_MAPPINGS.update(_P)
    NODE_DISPLAY_NAME_MAPPINGS.update(_Q)
except Exception as exc:
    print(f"[MOSS多身份] 逐人身体遮罩节点加载失败，原因：{exc}")

# self_check.py 只提供函数，不注册节点：缺依赖由「人脸定位地盘」开工时静默自查兜底，
# 不再要求使用者先去找一个体检节点跑一遍。

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
