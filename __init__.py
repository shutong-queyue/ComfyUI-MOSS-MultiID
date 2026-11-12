"""ComfyUI-MOSS-MultiID

给 MOSS-Image 做多个人物各自生效、不串脸的辅助节点。
只新增本文件夹，不修改 ComfyUI 本体，不修改任何模型权重。
删除本文件夹即可完全还原。
"""

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

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
    from .person_masks import (NODE_CLASS_MAPPINGS as _P,
                               NODE_DISPLAY_NAME_MAPPINGS as _Q)
    NODE_CLASS_MAPPINGS.update(_P)
    NODE_DISPLAY_NAME_MAPPINGS.update(_Q)
except Exception as exc:
    print(f"[MOSS多身份] 逐人身体遮罩节点加载失败，原因：{exc}")

# self_check.py 只提供函数，不注册节点：缺依赖由「人脸定位地盘」开工时静默自查兜底，
# 不再要求使用者先去找一个体检节点跑一遍。

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
