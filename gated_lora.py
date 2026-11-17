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
                                   "tooltip": "这个滤镜加多重：\n"
                                              "· 调低不会让动作更多，只会让本人特征变淡甚至消失；\n"
                                              "· 调高过头会糊、会串到别人身上。\n"
                                              "不同 LoRA、不同底模合适的值不一样，先从默认值试，再按现象微调。\n"
                                              "想换更像的人，先检查地盘对不对，而不是先动这里。"}),
                "文字部分也生效": (["否", "是"], {"default": "否",
                    "tooltip": "滤镜要不要也作用在文字对应的部分：\n"
                               "· 否＝只管画面格子，视角、动作这类描述保留更完整，推荐默认；\n"
                               "· 是＝人物特征略浓，但抽象描述（俯视、多只手、镜头角度）容易被抹平，"
                               "画面也更容易变暗。\n"
                               "两者差别在带位置信息的输入上更明显，纯长句描述时差距会小一些。"}),
            },
            "optional": {
                "清单": ("LORA_LIST", {"tooltip": "把上一个分区滤镜的『清单』接进来，"
                                                 "这里就会把「谁挂了哪个滤镜」一路记下去。\n"
                                                 "接上以后，后面『按清单挂滤镜』只要一个节点，"
                                                 "就能按编号把每个人的滤镜补到自己的裁片上，"
                                                 "不用给每个人再搭一个滤镜节点。"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING", "LORA_LIST")
    RETURN_NAMES = ("模型", "说明", "清单")
    FUNCTION = "apply"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "只在指定地盘起作用的人物滤镜；可串多个，每人管自己的区域"

    def apply(self, 模型, 地盘数据, 滤镜文件, 人物编号, 强度, 文字部分也生效, 清单=None):
        import folder_paths
        # 清单：一路记下"几号挂了哪个滤镜"。接下去给『按清单挂滤镜』用，
        # 这样小脸裁片那一段不用再逐个填编号。
        trace = [dict(e) for e in (清单 or [])]
        trace.append({"owner": int(人物编号), "lora": 滤镜文件, "strength": float(强度),
                      "text": 文字部分也生效 == "是"})
        boxes = 地盘数据["boxes"]
        hit = [i for i, b in enumerate(boxes) if int(b["owner"]) == int(人物编号)]
        if not hit:
            have = "、".join(str(b["owner"]) for b in boxes) or "（空）"
            note = (f"已跳过：地盘里没有编号 {人物编号} 的框（现有 {have}）。"
                    f"画面里人数比这个少，这个滤镜没挂上，模型原样传过去了。"
                    f"不用删节点——多出来的分区滤镜会自己跳过。")
            return (模型, note, trace)
        gi = hit[0]
        if 强度 <= 0.0:
            return (模型, f"强度为 0，编号 {人物编号} 未挂任何滤镜，模型原样传过去了。", trace)

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

        # 裁片模式：一批里每张裁片是"一个人头"，各归各的编号。
        # 这时门控要按批内第几张分别算，不能像整幅画布那样一套权重铺满全图。
        per_item = bool(地盘数据.get("按项门控"))
        owners = list(地盘数据.get("项归属") or [])
        sel = []
        if per_item:
            sel = [i for i, k in enumerate(owners) if int(k) == int(人物编号)]
            if not sel:
                have = "、".join(str(k) for k in owners) or "（空）"
                return (模型, f"已跳过：这一批裁片里没有编号 {人物编号}（现有 {have}）。"
                              f"这个滤镜没挂上，模型原样传过去了。", trace)

        cache = {}

        def gate(total_len, device, dtype, batch):
            key = (device, dtype, batch)
            g = cache.get(key)
            if g is None or g.shape[1] != total_len:
                if total_len < n_img:
                    return None
                n_txt = total_len - n_img
                g = torch.zeros((batch, total_len, 1), device=device, dtype=dtype)
                if per_item:
                    w_all = 地盘数据["weight"].detach().to(device=device, dtype=dtype)
                    for i in sel:
                        if i < batch:
                            g[i, :n_img, 0] = w_all[i]
                else:
                    g[:, :n_img, 0] = w_img.to(device=device, dtype=dtype)
                if text_full:
                    g[:, n_img:, 0] = 1.0
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
                g = gate(x.shape[1], x.device, x.dtype, x.shape[0])
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

        if per_item:
            share = float(sum(float(地盘数据["weight"][i].sum()) for i in sel))
            info = (f"已挂：{滤镜文件}\n这一批裁片里编号 {人物编号} 占 {len(sel)} 张"
                    f"｜强度 {强度:.2f}｜融合写法 {'是' if fused else '否'}"
                    f"｜挂上 {len(tensors)} 个位置｜文字部分{'全额' if text_full else '不'}生效"
                    f"｜份额合计 {share:.1f} 格")
            return (m, info, trace)

        _hint = ("★ 接线：后面还串了别人的滤镜就接到下一个；没有了就务必接到第二趟采样器的『model』上，"
                 "否则这个滤镜一次都不参与生成。")
        info = (_hint + "\n" + f"已挂：{滤镜文件}\n人物编号 {人物编号}（{boxes[gi]['name']}）"
                f"｜强度 {强度:.2f}｜融合写法 {'是' if fused else '否'}"
                f"｜挂上 {len(tensors)} 个位置｜文字部分{'全额' if text_full else '不'}生效"
                f"｜地盘份额 {float(w_img.sum()):.1f}/{n_img} 格")
        return (m, info, trace)




class MOSSIdLoraByList:
    """按清单挂滤镜：一个节点，把每个人的滤镜补到他自己那张裁片上

    小脸那一段要"每张小脸各挂各的滤镜"，如果按老办法就得给每个人再搭一个
    分区滤镜节点、再填一遍编号。七个人就是七个节点七次填号，很容易对不上。
    所以让上面的分区滤镜把"几号挂了什么"记成清单，这里读清单一次挂完——
    编号自动沿用上一次，不用手填。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL", {"tooltip": "接没挂人物滤镜的底模（一般就是接完加速滤镜那个）"}),
                "地盘数据": ("REGION_MAP", {"tooltip": "接『小脸裁剪放大』的『裁片地盘』。"
                                                    "每张裁片属于几号，从这里读"}),
                "清单": ("LORA_LIST", {"tooltip": "接最后一个『人物滤镜·分区版』的『清单』。"
                                                "里面记着各组'编号 → 滤镜文件'，"
                                                "和上一次（第二趟）用的是同一套"}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("模型", "说明")
    FUNCTION = "apply"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "按编号清单，一次给所有裁片各挂各的滤镜；编号自动沿用，不用手填"

    def apply(self, 模型, 地盘数据, 清单):
        import os as _os
        import folder_paths
        entries = {}
        for e in (清单 or []):
            if not e.get("lora"):
                continue
            entries[int(e.get("owner") or 0)] = e      # 同编号后来者覆盖
        if not entries:
            return (模型, "清单是空的（上面没接『人物滤镜·分区版』的『清单』，或者滤镜文件没选）"
                          "→ 模型原样传过去，这一批裁片就只按原话重画。")

        boxes = 地盘数据["boxes"]
        owners = [int(v) for v in (地盘数据.get("项归属") or [])]
        per_item = bool(地盘数据.get("按项门控"))
        n_img = int(地盘数据["n_img"])
        have = set(owners) if per_item else set(int(b["owner"]) for b in boxes)
        active = {k: v for k, v in entries.items() if k in have}
        if not active:
            name = "、".join(str(k) for k in sorted(have)) or "（空）"
            return (模型, "这一批裁片里的编号（%s）都不是清单里的（清单：%s）"
                          "→ 模型原样传过去，这次没挂滤镜。"
                    % (name, "、".join(str(k) for k in sorted(entries))))

        dm = 模型.model.diffusion_model
        plan = []          # [(owner, 滤镜文件, {(层,投影): (pairs, scale)})]
        fused_any = False
        for k, e in sorted(active.items()):
            path = folder_paths.get_full_path("loras", e["lora"])
            if not path or not _os.path.exists(path):
                raise ValueError("找不到滤镜文件：" + str(e["lora"]))
            table = read_lora(path)
            if not table:
                raise ValueError("这个滤镜文件里没读到任何 MOSS 层：" + str(e["lora"]))
            groups, fused = _resolve_targets(dm, table)
            fused_any = fused_any or fused
            m_t = {}
            for (idx, target), items in groups.items():
                pairs, scale = [], 1.0
                for part, slot in items:
                    A, B = slot["A"], slot["B"]
                    rank = A.shape[0]
                    scale = slot.get("alpha", float(rank)) / max(1, rank)
                    pairs.append((A, B))
                m_t[(idx, target)] = (pairs, scale * float(e.get("strength") or 1.3))
            plan.append((k, e["lora"], m_t))
        if not plan:
            return (模型, "清单里的滤镜一个都没挂上（层名和模型对不上）→ 原样传过去。")

        # 每层的钩子：把所有编号的增量按各自的格子加进去
        by_mod = {}
        for k, lora, m_t in plan:
            for key, (pairs, scale) in m_t.items():
                by_mod.setdefault(key, []).append((k, pairs, scale))

        w_all = 地盘数据["weight"].detach()
        cache = {}

        def gate_for(owner, total_len, device, dtype, batch):
            key = (owner, device, dtype, batch, total_len)
            g = cache.get(key)
            if g is not None:
                return g
            if total_len < n_img:
                return None
            g = torch.zeros((batch, total_len, 1), device=device, dtype=dtype)
            w = w_all.to(device=device, dtype=dtype)
            if per_item:
                rows = [i for i, o in enumerate(owners) if o == owner]
                for i in rows:
                    if i < batch:
                        g[i, :n_img, 0] = w[i]
            else:
                row = None
                for j, b in enumerate(boxes):
                    if int(b["owner"]) == owner:
                        row = j
                        break
                if row is not None and row < w.shape[0]:
                    g[:, :n_img, 0] = w[row]
            ent = next((e for e in plan if e[0] == owner), None)
            if ent is not None and (清单 or [{}])[-1].get("text"):
                pass
            cache[key] = g
            return g

        def make_hook(entries_here):
            def hook(mod, inputs, output):
                x = inputs[0]
                if not isinstance(x, torch.Tensor) or x.dim() != 3:
                    return output
                total = 0.0
                for owner, pairs, scale in entries_here:
                    g = gate_for(owner, x.shape[1], x.device, x.dtype, x.shape[0])
                    if g is None:
                        continue
                    acc = []
                    for A, B in pairs:
                        Ad = A.to(device=x.device, dtype=x.dtype)
                        Bd = B.to(device=x.device, dtype=x.dtype)
                        acc.append((x @ Ad.t()) @ Bd.t())
                    delta = acc[0] if len(acc) == 1 else torch.cat(acc, dim=-1)
                    total = total + delta * (scale * g)
                return output + total if isinstance(total, torch.Tensor) else output
            return hook

        def wrapper(executor, *args, **kwargs):
            dmod = executor.class_obj
            handles = []
            for (idx, target), entries_here in by_mod.items():
                try:
                    mod = dmod.layers[idx].proj[target]
                except Exception:
                    continue
                handles.append(mod.register_forward_hook(make_hook(entries_here)))
            try:
                return executor(*args, **kwargs)
            finally:
                for h in handles:
                    h.remove()

        m = 模型.clone()
        m.model_options.setdefault("transformer_options", {})
        comfy.patcher_extension.add_wrapper(WrappersMP.DIFFUSION_MODEL, wrapper,
                                            m.model_options, is_model_options=True)
        rows = []
        for k, lora, m_t in plan:
            cnt = len([i for i, o in enumerate(owners) if o == k]) if per_item else \
                (1 if any(int(b["owner"]) == k for b in boxes) else 0)
            rows.append("%d号 → %s（%d 张裁片）" % (k, _os.path.basename(lora), cnt))
        info = ("按清单挂好 %d 个人的滤镜：\n  " % len(plan) + "\n  ".join(rows)
                + "\n融合写法 %s｜挂上 %d 个位置" % ("是" if fused_any else "否", len(by_mod)))
        return (m, info)


NODE_CLASS_MAPPINGS_EXTRA = {"MOSSIdGatedLora": MOSSIdGatedLora,
                            "MOSSIdLoraByList": MOSSIdLoraByList}
NODE_DISPLAY_NAME_MAPPINGS_EXTRA = {"MOSSIdGatedLora": "人物滤镜·分区版",
                                    "MOSSIdLoraByList": "按清单挂滤镜"}
