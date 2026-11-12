"""人物滤镜·分区版（P1 阶段）

给 MOSS-Image 挂人物滤镜（LoRA），但只在指定的那块地盘上起作用。
做法：不把滤镜并进模型权重，而是每次计算时单独补一笔，补多少由"地盘份额"决定。
不修改 ComfyUI 本体，不修改模型权重文件。
"""

import os
import re

import torch

import comfy.patcher_extension
from comfy.patcher_extension import WrappersMP

try:
    from safetensors import safe_open
except Exception:  # 环境缺库时给出中文提示，而不是静默失效
    safe_open = None

_KEY_RE = re.compile(
    r"^(?:lora_unet[._])?layers[._](\d+)[._]proj[._](q|k|v|o|ffn_up_gate|ffn_down)[._](.+)$"
)
_PROJ_ORDER = ("q", "k", "v")


def _classify(rest):
    base = rest.replace(".weight", "")
    if base == "alpha":
        return "alpha"
    if base in ("lora_down", "lora_A"):
        return "A"
    if base in ("lora_up", "lora_B"):
        return "B"
    return None


def read_lora(path):
    """读滤镜文件，返回 {(层号, 投影名): {"A":..,"B":..,"alpha":..}}"""
    if safe_open is None:
        raise RuntimeError("当前 Python 环境里没有 safetensors 库，读不了滤镜文件。")
    out = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            m = _KEY_RE.match(k.replace("diffusion_model.", "").replace("diffusion_model_", ""))
            if not m:
                continue
            idx, part, rest = int(m.group(1)), m.group(2), m.group(3)
            kind = _classify(rest)
            if kind is None:
                continue
            slot = out.setdefault((idx, part), {})
            t = f.get_tensor(k)
            if kind == "alpha":
                slot["alpha"] = float(t.reshape(-1)[0]) if t.numel() else 0.0
            else:
                slot[kind] = t.float()
    return out


def _resolve_targets(dm, table):
    """按模型实际情况决定：融合版把 q/k/v 合成一个 qkv，分开版各自挂。"""
    first = None
    for (idx, part) in table:
        if idx == 0:
            first = part
            break
    probe = dm.layers[0].proj
    fused = ("qkv" in probe) and ("q" not in probe)
    groups = {}
    for (idx, part), slot in table.items():
        if "A" not in slot or "B" not in slot:
            continue
        if fused and part in _PROJ_ORDER:
            target = "qkv"
        else:
            target = part
        groups.setdefault((idx, target), []).append((part, slot))
    ordered = {}
    for key, items in groups.items():
        if key[1] == "qkv":
            items = sorted(items, key=lambda p: _PROJ_ORDER.index(p[0]))
        ordered[key] = items
    return ordered, fused


class MOSSIdGatedLora:
    """人物滤镜·分区版"""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {
            "required": {
                "模型": ("MODEL", {"tooltip": "上一个人的输出，串起来即可：第 1 个接全局滤镜，"
                                             "第 2 个接第 1 个的『模型』输出"}),
                "地盘数据": ("REGION_MAP", {"tooltip": "来自『人脸定位地盘』或『地盘分配器』，"
                                                    "决定这个滤镜只在哪些格子里起作用"}),
                "滤镜文件": (sorted(folder_paths.get_filename_list("loras")), {
                    "tooltip": "这个人物对应的 LoRA 权重。换人就是换这里"}),
                "人物编号": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1,
                                     "tooltip": "要作用在几号人身上，对应地盘节点里的 owner 编号"
                                                "（默认按画面位置排序：最左＝1 号）。\n"
                                                "地盘里没有这个编号时，本节点会自动跳过并把模型原样传过去。"}),
                "强度": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 2.5, "step": 0.05,
                                   "tooltip": "这个滤镜加多重，默认 1.3 是常用起点：\n"
                                              "· 调低不会让动作更多，只会让本人特征变淡甚至消失；\n"
                                              "· 调高过头会糊、会串到别人身上。\n"
                                              "想换更像的人，先检查地盘对不对，而不是先动这里。"}),
                "文字部分也生效": (["否", "是"], {"default": "否",
                    "tooltip": "滤镜要不要也作用在文字对应的部分：\n"
                               "· 否＝只管画面格子，视角、动作这类描述保留更完整，推荐默认；\n"
                               "· 是＝人物特征略浓，但抽象描述（俯视、多只手、镜头角度）容易被抹平，"
                               "画面也更容易变暗。\n"
                               "两者差别在带位置信息的输入上更明显，纯长句描述时差距会小一些。"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("模型", "说明")
    FUNCTION = "apply"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "只在指定地盘起作用的人物滤镜；可串多个，每人管自己的区域"

    def apply(self, 模型, 地盘数据, 滤镜文件, 人物编号, 强度, 文字部分也生效):
        import folder_paths
        boxes = 地盘数据["boxes"]
        hit = [i for i, b in enumerate(boxes) if int(b["owner"]) == int(人物编号)]
        if not hit:
            have = "、".join(str(b["owner"]) for b in boxes) or "（空）"
            note = (f"已跳过：地盘里没有编号 {人物编号} 的框（现有 {have}）。"
                    f"画面里人数比这个少，这个滤镜没挂上，模型原样传过去了。"
                    f"不用删节点——多出来的分区滤镜会自己跳过。")
            return (模型, note)
        gi = hit[0]
        if 强度 <= 0.0:
            return (模型, f"强度为 0，编号 {人物编号} 未挂任何滤镜，模型原样传过去了。")

        abs_path = folder_paths.get_full_path("loras", 滤镜文件)
        if not abs_path or not os.path.exists(abs_path):
            raise ValueError(f"找不到滤镜文件：{滤镜文件}")
        table = read_lora(abs_path)
        if not table:
            raise ValueError("这个滤镜文件里没读到任何 MOSS 层（键名格式不认识）。")

        dm = 模型.model.diffusion_model
        groups, fused = _resolve_targets(dm, table)
        if not groups:
            raise ValueError("滤镜里的层和模型对不上，一个都没挂上。")

        w_img = 地盘数据["weight"][gi].detach()
        n_img = int(地盘数据["n_img"])
        text_full = (文字部分也生效 == "是")

        cache = {}

        def gate(total_len, device, dtype):
            key = (device, dtype)
            g = cache.get(key)
            if g is None or g.shape[1] != total_len:
                if total_len < n_img:
                    return None
                n_txt = total_len - n_img
                head = w_img.to(device=device, dtype=dtype)
                tail = (torch.ones if text_full else torch.zeros)(
                    n_txt, device=device, dtype=dtype)
                g = torch.cat([head, tail]).view(1, total_len, 1)
                cache[key] = g
            return g

        tensors = {}
        for (idx, target), items in groups.items():
            pairs = []
            scale = 1.0
            for part, slot in items:
                A, B = slot["A"], slot["B"]
                rank = A.shape[0]
                alpha = slot.get("alpha", float(rank))
                scale = alpha / max(1, rank)
                pairs.append((A, B))
            tensors[(idx, target)] = (pairs, scale * 强度)

        def make_hook(pairs, scale):
            def hook(mod, inputs, output):
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.dim() != 3:
                    return output
                g = gate(x.shape[1], x.device, x.dtype)
                if g is None:
                    return output
                acc = []
                for A, B in pairs:
                    Ad = A.to(device=x.device, dtype=x.dtype)
                    Bd = B.to(device=x.device, dtype=x.dtype)
                    acc.append((x @ Ad.t()) @ Bd.t())
                delta = acc[0] if len(acc) == 1 else torch.cat(acc, dim=-1)
                return output + delta * (scale * g)
            return hook

        def wrapper(executor, *args, **kwargs):
            dmod = executor.class_obj
            handles = []
            for (idx, target), (pairs, scale) in tensors.items():
                try:
                    mod = dmod.layers[idx].proj[target]
                except Exception:
                    continue
                handles.append(mod.register_forward_hook(make_hook(pairs, scale)))
            try:
                return executor(*args, **kwargs)
            finally:
                for h in handles:
                    h.remove()

        m = 模型.clone()
        m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper(WrappersMP.DIFFUSION_MODEL, wrapper,
                                            m.model_options, is_model_options=True)

        info = (f"已挂：{滤镜文件}\n人物编号 {人物编号}（{boxes[gi]['name']}）"
                f"｜强度 {强度:.2f}｜融合写法 {'是' if fused else '否'}"
                f"｜挂上 {len(tensors)} 个位置｜文字部分{'全额' if text_full else '不'}生效"
                f"｜地盘份额 {float(w_img.sum()):.1f}/{n_img} 格")
        return (m, info)


NODE_CLASS_MAPPINGS_EXTRA = {"MOSSIdGatedLora": MOSSIdGatedLora}
NODE_DISPLAY_NAME_MAPPINGS_EXTRA = {"MOSSIdGatedLora": "人物滤镜·分区版"}
