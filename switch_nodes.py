"""流程开关（两个小节点）

1) 环节开关（图）——给 A 方案工作流用：
   把"这一环节的成品"和"上一环节的成品"接进来，开关一拨就只跑选中的那条支路。
   两个图像口都是懒加载：没选中的那一支 ComfyUI 根本不会执行，
   所以关掉②就不会白跑第二趟采样，关掉③就不会白裁小脸。

2) 遮罩来源——把『人脸定位地盘』的『遮罩来源』变成一个可以拨的开关：
   自动切人（本节点自己调 SAM）/ 外部遮罩（用你接进来的那一块）/ 只按脸就近。
"""


class MOSSIdStageSwitch:
    """环节开关：只跑选中的那一支"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "开关": ("BOOLEAN", {"default": True,
                    "tooltip": "开＝走『开时』那一支（例如：跑这一环节）；关＝走『关时』那一支（例如：跳过，直接用上一环节的图）。\n"
                               "『条件』为否时也会走『关时』。"}),
            },
            "optional": {
                "条件": ("BOOLEAN", {"default": True,
                    "tooltip": "可选。接“这一环节有没有活干”的判断（地盘的『有脸』、小脸的『有裁片』）。\n"
                               "条件＝否时直接走『关时』，『开时』整条支路不执行。不接＝只看开关。"}),
                "关时": ("IMAGE", {"lazy": True,
                    "tooltip": "开关＝关时输出的图（通常是上一环节的成品）。"}),
                "开时": ("IMAGE", {"lazy": True,
                    "tooltip": "开关＝开时输出的图（这一环节的成品）。"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图",)
    FUNCTION = "run"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "二选一：只执行被选中的那一支，没选中的整条支路都不会跑"

    def check_lazy_status(self, 开关, 条件=True, 关时=None, 开时=None):
        return ["开时"] if (开关 and 条件) else ["关时"]

    def run(self, 开关, 条件=True, 关时=None, 开时=None):
        on = bool(开关) and bool(条件)
        img = 开时 if on else 关时
        if img is None:
            raise RuntimeError("环节开关：这一支没接图（开关＝%s，条件＝%s）。" % (开关, 条件))
        return (img,)


class MOSSIdMaskSource:
    """遮罩来源开关：给『人脸定位地盘』的『来源开关』口一个可拨的值"""

    MODES = {
        "自动切人（本节点调 SAM）": "自动逐人切（本节点内置调用）",
        "外部遮罩（接『遮罩』口）": "外部接入（下面『遮罩』口）",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "来源": (list(cls.MODES.keys()), {"default": "自动切人（本节点调 SAM）",
                    "tooltip": "接在『人脸定位地盘』的『来源开关』口上：\n"
                               "· 自动切人＝地盘节点自己调 SAM，把每个人各切一块；\n"
                               "· 外部遮罩＝用你接到『遮罩』口的那块（例如 BiRefNet 的人像剪影）；\n"
                               "选自动时只是不使用外部那块（外部节点仍会计算，结果被忽略）。"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("来源值",)
    FUNCTION = "run"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "把『遮罩来源』做成一个可拨的开关，不用每次进节点改下拉"

    def run(self, 来源):
        return (self.MODES.get(来源, self.MODES["自动切人（本节点调 SAM）"]),)


NODE_CLASS_MAPPINGS = {"MOSSIdStageSwitch": MOSSIdStageSwitch,
                       "MOSSIdMaskSource": MOSSIdMaskSource}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSIdStageSwitch": "环节开关（图）",
                              "MOSSIdMaskSource": "遮罩来源开关"}
