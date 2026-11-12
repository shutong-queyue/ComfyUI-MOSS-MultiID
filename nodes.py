"""地盘分配器 + 地盘预览图（P0 阶段）

作用：把你画的方框，换算成模型真正使用的「小格子」归属表，并画成一张彩色图给你核对。
本阶段完全不碰模型、不碰注意力、不碰权重。只是算术 + 画图。

卸载方法：删掉本文件夹即可，ComfyUI 本体未做任何修改。
"""

import json

import numpy as np
import torch

DS = 16  # 1 个小格子 = 16x16 像素（依据 latent_formats.Flux2 的 spacial_downscale_ratio=16）
_SAM2_CKPT = "sam2.1_hiera_base_plus.pt"
_PM = None


def _person_masks_mod():
    """按需加载同插件里的 SAM2 模块（ComfyUI 把插件当包导入，绝对导入会失败）"""
    global _PM
    if _PM is None:
        try:
            from . import person_masks as _pm
        except Exception:
            import importlib.util
            import os as _os
            spec = importlib.util.spec_from_file_location(
                "person_masks", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "person_masks.py"))
            _pm = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_pm)
        _PM = _pm
    return _PM


def _pool_mask(mm, gh, gw):
    """把 (N,H,W) 的遮罩按块平均降到网格 (N, gh*gw)，并可选胀开"""
    import cv2
    ys = np.linspace(0, mm.shape[1], gh + 1).astype(int)
    xs = np.linspace(0, mm.shape[2], gw + 1).astype(int)
    return np.stack([[[float(mm[j, a:b, c:d].mean()) for c, d in zip(xs[:-1], xs[1:])]
                      for a, b in zip(ys[:-1], ys[1:])]
                     for j in range(mm.shape[0])], dtype=np.float32).reshape(mm.shape[0], -1)


def _dilate(mm, pixels):
    if pixels <= 0:
        return mm
    import cv2
    k = 2 * max(1, int(round(pixels / 2.0))) + 1
    ker = np.ones((k, k), np.uint8)
    return np.stack([cv2.dilate(x, ker) for x in mm]).astype(np.float32)

PALETTE = [
    (0.20, 0.75, 0.45),
    (0.30, 0.55, 0.95),
    (0.95, 0.65, 0.20),
    (0.85, 0.40, 0.55),
    (0.60, 0.40, 0.90),
    (0.35, 0.80, 0.80),
    (0.90, 0.85, 0.35),
    (0.75, 0.45, 0.30),
]

DEFAULT_BOXES = (
    '[\n'
    '  {"owner": 1, "name": "人物一", "x": 140, "y": 140, "w": 520, "h": 800},\n'
    '  {"owner": 2, "name": "人物二", "x": 780, "y": 140, "w": 520, "h": 800}\n'
    ']'
)


def _norm_one(d, idx, canvas=None):
    """把各种来源的框统一成 {owner,name,x,y,w,h}；认不出来就报错并打印原始内容"""
    if not isinstance(d, dict):
        raise ValueError(f"第 {idx+1} 个框不是字典，实际收到：{repr(d)[:200]}")
    if "x" in d and ("width" in d or "w" in d):
        x = int(float(d["x"])); y = int(float(d.get("y", 0)))
        w = int(float(d.get("width", d.get("w", 1)))); h = int(float(d.get("height", d.get("h", 1))))
    elif all(k in d for k in ("x1", "x2", "y1", "y2")):
        x = int(float(d["x1"])); y = int(float(d["y1"]))
        w = max(1, int(float(d["x2"])) - x); h = max(1, int(float(d["y2"])) - y)
    elif "bbox" in d and isinstance(d["bbox"], (list, tuple)) and len(d["bbox"]) == 4:
        b = [float(v) for v in d["bbox"]]
        y0, x0, y1, x1 = b                      # 官方顺序是 yxyx
        if min(b) < 0:
            if not canvas:
                raise ValueError("bbox 是负数（以画布中心为原点），但没给画布尺寸，无法换算成像素")
            cx, cy = canvas[0] / 2.0, canvas[1] / 2.0
            x0 += cx; x1 += cx; y0 += cy; y1 += cy
        x, y = int(x0), int(y0)
        w = max(1, int(x1 - x0)); h = max(1, int(y1 - y0))
    else:
        raise ValueError(f"第 {idx+1} 个框认不出坐标，收到的原始内容是：{repr(d)[:400]}")
    nm = d.get("name") or d.get("label") or d.get("caption") or d.get("desc") or ""
    nm = str(nm)[:24] if nm else f"{idx+1}号"
    one = {"owner": int(d.get("owner", idx + 1)), "name": nm, "x": x, "y": y, "w": w, "h": h}
    # 脸位点（可选）：有它才能做「脸不许被交界线穿过」的保护
    try:
        one["hx"] = int(float(d.get("hx", d.get("head_x"))))
        one["hy"] = int(float(d.get("hy", d.get("head_y"))))
    except (TypeError, ValueError):
        one["hx"] = one["hy"] = None
    one["head_src"] = str(d.get("head_src", ""))[:16]
    return one


def _parse_boxes(boxes_text, boxes_link, canvas=None):
    """把「手打的方框」和「画框工具连进来的框」合并成统一列表。

    画框工具的框支持多种来源：官方 Create Bounding Boxes / MOSS-Image Prompt Builder KJ
    输出的 BOUNDING_BOX、ARRAY，或普通 BOUNDINGBOX，也可以是 JSON 文本。
    """
    items = []
    if boxes_link is not None:
        boxes_text = ""          # 连了线就只认线进来的，避免和手写内容重复
    if boxes_text and str(boxes_text).strip():
        try:
            data = json.loads(boxes_text)
        except Exception as exc:
            raise ValueError(f"方框内容格式不对，请检查引号和逗号。具体报错：{exc}")
        if isinstance(data, dict):
            data = [data]
        for i, d in enumerate(data):
            one = _norm_one(d, i, canvas)
            one["owner"] = int(d.get("owner", i + 1))
            if d.get("name"):
                one["name"] = str(d["name"])
            items.append(one)
    if boxes_link is not None:
        raw = boxes_link
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raise ValueError(f"连进来的框是文本但解析不了，原始内容：{raw[:200]}")
        if isinstance(raw, dict):
            raw = [raw]
        for i, one in enumerate(raw):
            items.append(_norm_one(one, len(items), canvas))
    if not items:
        raise ValueError("一个方框都没有。请在方框内容里填，或者从画框工具连一根线过来。")
    if len(items) > 12:
        raise ValueError(f"目前最多支持 12 个人物，你填了 {len(items)} 个框。")
    return items


def _signed_distance(px, py, box):
    """格子中心到方框边的带符号距离：框内为正（越往里越大），框外为负。"""
    x0, y0 = box["x"], box["y"]
    x1, y1 = x0 + box["w"], y0 + box["h"]
    dx = np.minimum(px - x0, x1 - px)
    dy = np.minimum(py - y0, y1 - py)
    inside = (dx >= 0) & (dy >= 0)
    return np.where(inside, np.minimum(dx, dy), -np.maximum(-dx, -dy))


class MOSSIdRegionMap:
    """地盘分配器：方框 → 每个小格子属于谁"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "画布宽": ("INT", {"default": 1440, "min": 64, "max": 8192, "step": DS, "tooltip": '画面宽度（像素）。要和实际出图用的画布尺寸一致，否则格子会错位；建议取 16 的倍数。'}),
                "画布高": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": DS, "tooltip": '画面高度（像素）。要和实际出图用的画布尺寸一致；不是 16 的倍数时，最后几行会不属于任何人。'}),
                "方框内容": ("STRING", {"multiline": True, "default": DEFAULT_BOXES,
                                       "tooltip": "每行一个框：owner 是编号，x/y 是左上角像素，w/h 是宽高。\n"
                                                  "能加上 hx,hy（这个人的脸位点）更准——"
                                                  "遮罩模式靠它认人，脸保护圈也靠它。"}),
                "边界抹软像素": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1,
                                         "tooltip": "0＝硬切；越大，边界处几个人按比例分摊"}),
                "重叠时归谁": (["就近", "先填的优先"], {"default": "就近",
                                              "tooltip": "两个框压在一起时给谁：就近＝离谁的框近给谁；"
                                                         "先填的优先＝按列表顺序"}),
                "遮罩用法": (["不用遮罩（纯按方框）",
                            "整幅人像剪影（只区分人和背景）",
                            "每人一块（接外部遮罩）",
                            "每人一块（SAM2 自动切，要连参考图）"],
                           {"default": "不用遮罩（纯按方框）",
                            "tooltip": "手动框也能叠加遮罩，四种任选：\n"
                                       "· 不用＝完全按你画的框划分；\n"
                                       "· 整幅剪影＝背景一格滤镜都不给；\n"
                                       "· 每人一块（外部）＝一人一张遮罩；\n"
                                       "· 每人一块（SAM2 自动）＝节点看着参考图自己切，"
                                       "认人靠框里的脸位点。\n"
                                       "共同规则：遮罩盖到的格子按遮罩判，没盖到的格子仍按方框判，"
                                       "所以遮罩漏切也不会出现没人管的一片。"}),
                "遮罩放大像素": ("INT", {"default": 0, "min": 0, "max": 192, "step": 4,
                                      "tooltip": "遮罩向外胀多少像素，补回发梢、衣角、肩膀。"
                                                 "0 最保守，96 以上容易和邻居连成一片"}),
            },
            "optional": {
                "画框工具的框": ("BOUNDINGBOX,BOUNDING_BOX,ARRAY,STRING", {
                    "tooltip": "可选：把画框工具输出的框直接连过来。一连上就以它为准，"
                               "『方框内容』里手写的会被忽略。"}),
                "遮罩": ("MASK", {
                    "tooltip": "接 1 张＝整幅人像剪影；接多张＝一人一块。"
                               "选『不用遮罩』或『SAM2 自动切』时可以不接"}),
                "参考图": ("IMAGE", {
                    "tooltip": "只有『每人一块（SAM2 自动切）』需要：把要看的那张图连进来"}),
            },
        }

    RETURN_TYPES = ("REGION_MAP", "STRING")
    RETURN_NAMES = ("地盘数据", "统计说明")
    FUNCTION = "build"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "把方框换算成小格子归属表；可选叠加人像剪影／每人一块遮罩（外部或 SAM2 自动切）"

    def build(self, 画布宽, 画布高, 方框内容, 边界抹软像素, 重叠时归谁, 遮罩用法, 遮罩放大像素,
               画框工具的框=None, 遮罩=None, 参考图=None):
        gw = max(1, int(画布宽) // DS)
        gh = max(1, int(画布高) // DS)
        n_img = gw * gh
        items = _parse_boxes(方框内容, 画框工具的框, (int(画布宽), int(画布高)))
        k = len(items)

        col_c = (np.arange(gw, dtype=np.float32) + 0.5) * DS
        row_c = (np.arange(gh, dtype=np.float32) + 0.5) * DS
        CX, CY = np.meshgrid(col_c, row_c)
        px, py = CX.ravel(), CY.ravel()

        # 每个人的「深浅分数」：框内为正、框外为负
        scores = np.stack([_signed_distance(px, py, b) for b in items], axis=0).astype(np.float32)

        # ── 遮罩（手动版也能用，四种模式随便挑）─────────────────────────────
        # 原则：遮罩盖到的地方按遮罩判，没盖到的地方仍退回方框就近 —— 两套思路并存，不互相顶掉
        mask_note = ""
        用法 = str(遮罩用法)
        if 用法 != "不用遮罩（纯按方框）":
            try:
                if 用法.startswith("每人一块（SAM2 自动"):
                    if 参考图 is None:
                        raise ValueError("这个模式要把图连到『参考图』上（SAM2 要看着图切）")
                    pm = _person_masks_mod()
                    pts = []
                    for b in items:
                        cx = float(b.get("hx") if b.get("hx") is not None else b["x"] + b["w"] / 2.0)
                        cy = float(b.get("hy") if b.get("hy") is not None else b["y"] + b["h"] / 6.0)
                        w = max(24.0, float(b["w"]) * 0.35)
                        pts.append({"cx": cx, "cy": cy, "w": w, "h": w * 1.25})
                    rgb = np.clip(参考图[0].float().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                    got = pm.predict_person_masks(rgb, pts, _SAM2_CKPT, "cuda")
                    if not got:
                        raise ValueError("SAM2 一块都没切出来")
                    mm = np.stack(got).astype(np.float32)
                    mask_note = "遮罩来源：SAM2 自动切 %d 块（认人靠方框里的脸位点）" % mm.shape[0]
                else:
                    if 遮罩 is None:
                        raise ValueError("这个模式要接『遮罩』")
                    mm = 遮罩.detach().cpu().float().numpy()
                    if mm.ndim == 2:
                        mm = mm[None]
                    mask_note = "遮罩来源：外部接入 %d 张" % mm.shape[0]
                mm = _dilate(mm, int(遮罩放大像素))
                grid = _pool_mask(mm, gh, gw)
                if 用法.startswith("整幅人像剪影"):
                    sil = np.clip(grid[0], 0.0, 1.0)
                    scores = np.where(sil[None, :] > 0.5, scores, -999.0)
                    mask_note += "｜按剪影切背景：身上 %d 格（%.0f%%），背景 %d 格不发任何滤镜" % (
                        int((sil > 0.5).sum()), 100.0 * (sil > 0.5).mean(), int((sil <= 0.5).sum()))
                else:
                    own = np.zeros((k, n_img), dtype=np.float32)
                    got_n = 0
                    for i, b in enumerate(items):
                        cx = float(b.get("hx") if b.get("hx") is not None else b["x"] + b["w"] / 2.0)
                        cy = float(b.get("hy") if b.get("hy") is not None else b["y"] + b["h"] / 2.0)
                        gy = int(min(gh - 1, max(0, cy // DS)))
                        gx = int(min(gw - 1, max(0, cx // DS)))
                        gi = gy * gw + gx
                        j = int(grid[:, gi].argmax())
                        if float(grid[j, gi]) > 0.2:
                            own[i] = grid[j]
                            got_n += 1
                        else:
                            same = [jj for jj in range(grid.shape[0]) if grid[jj].max() > 0.2]
                            if same:
                                jj = min(same, key=lambda t: float(np.hypot(
                                    (grid[t] * px).sum() / max(1e-6, grid[t].sum()) - cx,
                                    (grid[t] * py).sum() / max(1e-6, grid[t].sum()) - cy)))
                                own[i] = grid[jj]
                                got_n += 1
                    covered = own.max(axis=0) > 0.5
                    scores = np.where(covered[None, :],
                                      (np.clip(own, 0.0, 1.0) - 0.5) * 240.0, scores)
                    mask_note += "｜%d/%d 人有自己的块；盖到处 %d 格按块判，没盖到的 %d 格仍按方框就近" % (
                        got_n, k, int(covered.sum()), int(n_img - covered.sum()))
            except Exception as exc:
                mask_note = "⚠ 遮罩没用上（%s），这一趟纯按方框判" % str(exc)[:120]

        # 关键：额外加一个「不属于任何人」的隐含竞争者，分数恒为 0。
        # 这样两个框之外的公共区域不会被强行分摊给某个人。
        base_row = np.zeros((1, n_img), dtype=np.float32)
        pool = np.concatenate([scores, base_row], axis=0)

        if 边界抹软像素 > 0:
            t = float(max(1, 边界抹软像素))
            e = np.exp((pool - pool.max(axis=0, keepdims=True)) / t)
            share_all = e / np.clip(e.sum(axis=0, keepdims=True), 1e-8, None)
            weight = share_all[:k]
        else:
            weight = np.zeros_like(scores)
            if 重叠时归谁 == "先填的优先":
                inside = scores > 0
                has_owner = inside.any(axis=0)
                chosen = np.where(has_owner, inside.argmax(axis=0), k)
            else:
                chosen = pool.argmax(axis=0)
            pick = chosen < k
            weight[chosen[pick], np.arange(n_img)[pick]] = 1.0

        # ── 脸不许被交界线穿过 ────────────────────────────────────────────
        # 只要每个框带了脸位点(hx,hy)，就在每块脸周围画一个保护圈：圈内的格子
        # 硬归这个人，身体框也好、"不属于任何人"的兜底也好，都不许把脸抢走。
        # 半径 = 到最近另一个脸位点距离的 0.9 倍（夹在画布短边的 7%~45% 之间）。
        # 为什么放大到 0.9 倍而不是刚好一半：圈之间重叠的部分按「相对距离」判，
        # 谁离自己的脸近归谁，所以圈放大并不会互相抢脸，只会把交界线推到
        # 两个人头位置的中垂线上 —— 并排就竖着分、上下就横着分、斜着就斜着分，
        # 全自动，不预设方向，也不会压在任意一张脸上。
        # 身体框仍然管用：圈够不到的地方（脚、背景、画框边缘）照身体框判，
        # 所以两个人各自的高低大小一点没被抹平。
        heads = [(b.get("hx"), b.get("hy")) for b in items]
        have_heads = all(hx is not None and hy is not None for hx, hy in heads)
        short = float(min(int(画布宽), int(画布高)))
        radii = []
        prot_cells = [0] * k
        protected = np.zeros(n_img, dtype=bool)
        if have_heads:
            dist = np.stack([np.hypot(px - float(hx), py - float(hy))
                             for hx, hy in heads]).astype(np.float32)
            for i in range(k):
                if k == 1:
                    r = 0.30 * short
                else:
                    dmin = min(float(np.hypot(heads[i][0] - heads[j][0],
                                              heads[i][1] - heads[j][1]))
                               for j in range(k) if j != i)
                    r = min(0.45 * short, max(0.07 * short, 0.9 * dmin))
                radii.append(r)
            rr = np.array(radii, dtype=np.float32)[:, None]
            inside = dist <= rr
            protected = inside.any(axis=0)
            # 两个圈重叠时，谁相对更近归谁（用 dist/半径 比大小，圈大的占优势）
            ratio = np.where(inside, dist / rr, 1e9)
            win = ratio.argmin(axis=0)
            weight = weight.copy()
            weight[:, protected] = 0.0
            weight[win[protected], np.arange(n_img)[protected]] = 1.0
            prot_cells = [int((inside[i] & (win == i)).sum()) for i in range(k)]

        owner_idx = weight.argmax(axis=0).astype(np.int64)
        owner_share = weight.max(axis=0)
        nobody = (scores.max(axis=0) <= 0) & (~protected)
        owner_idx = np.where(nobody, -1, owner_idx).astype(np.int8)


        wt = torch.from_numpy(np.ascontiguousarray(weight))
        ot = torch.from_numpy(np.ascontiguousarray(owner_idx))

        lines = [
            f"画布 {画布宽}×{画布高} 像素 → 小格子 {gw}×{gh} = 共 {n_img} 格（每格 {DS}×{DS} 像素）",
            f"人物框 {k} 个｜边界抹软 {边界抹软像素} 像素｜重叠时归：{重叠时归谁}",
            f"遮罩用法：{用法}" + (f"｜胀开 {遮罩放大像素} 像素" if int(遮罩放大像素) else ""),
            mask_note or "遮罩来源：没用遮罩，纯按方框就近判",
            ("脸保护：开着（交界线不许穿进任何人的脸圈）" if have_heads
             else "脸保护：没开（框里没有脸位点，退回只按方框判）"),
            "",
        ]
        for i, b in enumerate(items):
            own = int((owner_idx == i).sum())
            share = float(owner_share[owner_idx == i].sum()) if own else 0.0
            inside = int((scores[i] > 0).sum())
            pr = ""
            if have_heads:
                pr = f"｜脸圈半径 {radii[i]:.0f}px 独占 {prot_cells[i]} 格"
            lines.append(
                f"{b['name']}（{b['owner']}号）：独占 {own} 格，实际份额 {share:.1f} 格，"
                f"框内共 {inside} 格，位置 x{b['x']} y{b['y']} 尺寸 {b['w']}×{b['h']}{pr}"
            )
        shared = int(((scores > 0).sum(axis=0) > 1).sum())
        public = int(nobody.sum())
        lines += ["", f"两人重叠的格子：{shared} 格｜不属于任何人的公共区域：{public} 格"]
        if int(画布宽) % DS or int(画布高) % DS:
            lines.append("⚠ 画布宽/高不是 16 的倍数（%s×%s）：最后 %d 行 / %d 列拿不到任何滤镜，"
                         "建议改成 %d×%d。" % (
                             画布宽, 画布高, int(画布高) - gh * DS, int(画布宽) - gw * DS,
                             int(画布宽) // DS * DS, int(画布高) // DS * DS))
        if shared == 0 and k > 1:
            lines.append("提示：框之间没有重叠，适合并排构图；拥抱这类动作应该让它们重叠。")
        if have_heads and k >= 2:
            dmin_all = min(float(np.hypot(heads[i][0] - heads[j][0], heads[i][1] - heads[j][1]))
                           for i in range(k) for j in range(k) if i != j)
            lines.append(
                f"最近的两张脸相距 {dmin_all:.0f} 像素｜保护半径最小 {min(radii):.0f} 像素｜"
                f"安全线：脸位点只要偏不到 {dmin_all / 2:.0f} 像素，这张脸就整块归她自己")
            srcs = {b.get("head_src", "") for b in items}
            if any("估" in s for s in srcs):
                lines.append("注意：有人的脸位点是按身体框估的（模型那条没交），保护圈可能偏，"
                             "看看翻译结果里有没有那条提示。")
        else:
            lines.append("注意：框里没有脸位点，交界线还是按方框边走，可能正好压在脸上。"
                         "接上「一句话转构图·API版」的『地盘方框』就会带上脸位点。")

        region = {
            "grid_w": gw, "grid_h": gh, "n_img": n_img, "ds": DS,
            "canvas": (int(画布宽), int(画布高)),
            "boxes": items, "weight": wt, "owner": ot,
            "feather": int(边界抹软像素), "mode": 重叠时归谁,
            "heads": heads if have_heads else None,
            "radii": radii if have_heads else None,
            "prot": protected.reshape(gh, gw) if have_heads else None,
        }
        return (region, "\n".join(lines))


class MOSSIdRegionPreview:
    """地盘预览图：把归属表画成彩色图，不花算力先看对不对"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "地盘数据": ("REGION_MAP", {"tooltip": '接『人脸定位地盘』或『地盘分配器』的输出，里面存着每一格归谁的信息。'}),
                "涂色浓度": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": '彩色覆盖的浓度。0＝只看原图，越大越看不清原图。'}),
            },
            "optional": {
                "底图": ("IMAGE", {"tooltip": "可选：把第一趟草稿连进来，直接看地盘和实际站位对不对得上"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("预览图",)
    FUNCTION = "draw"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "按小格子给每个人物上色，重叠区自动混色，不属于任何人的区域不上色"

    def draw(self, 地盘数据, 涂色浓度, 底图=None):
        gw, gh = 地盘数据["grid_w"], 地盘数据["grid_h"]
        W, H = 地盘数据["canvas"]
        weight = 地盘数据["weight"].float().cpu().numpy()

        if 底图 is not None:
            base = 底图[0:1].float().cpu().numpy().copy()
            if base.shape[1] != H or base.shape[2] != W:
                ys = np.linspace(0, base.shape[1] - 1, H).astype(np.int32)
                xs = np.linspace(0, base.shape[2] - 1, W).astype(np.int32)
                base = base[:, ys][:, :, xs]
            base = base.astype(np.float32)
        else:
            base = np.full((1, H, W, 3), 0.13, dtype=np.float32)

        small = weight.reshape(weight.shape[0], gh, gw)
        # 按比值映射，不用整数倍 repeat：画布尺寸不是 16 的倍数时整数倍会少几行，形状对不上
        ri = np.minimum((np.arange(H) * gh) // H, gh - 1)
        ci = np.minimum((np.arange(W) * gw) // W, gw - 1)
        up = small[:, ri][:, :, ci]

        overlay = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(weight.shape[0]):
            rgb = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
            m = np.clip(up[i], 0.0, 1.0)
            for c in range(3):
                overlay[..., c] += m * rgb[c]
        acc = np.clip(up.sum(axis=0), 0.0, 1.0)[..., None]
        overlay = overlay / np.clip(acc, 1e-6, None)

        # 把「脸保护圈」和「脸位点」描出来，让人一眼看见交界线到底走没走在脸上
        heads, radii = 地盘数据.get("heads"), 地盘数据.get("radii")
        acc_full = acc[..., 0] if acc.ndim == 3 else acc
        if heads and radii:
            ds = 地盘数据.get("ds", DS)
            col = (np.arange(gw) + 0.5) * ds
            row = (np.arange(gh) + 0.5) * ds
            CXs, CYs = np.meshgrid(col, row)
            for i, (hx, hy) in enumerate(heads):
                if hx is None or hy is None:
                    continue
                d = np.sqrt((CXs - float(hx)) ** 2 + (CYs - float(hy)) ** 2)
                ring = ((np.abs(d - radii[i]) <= 1.2 * ds) | (d <= ds)).astype(np.float32)
                ru = ring[ri][:, ci]
                rgb = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32) * 0.55 + 0.45
                on = ru > 0
                for c in range(3):
                    overlay[..., c] = np.where(on, np.clip(rgb[c], 0, 1), overlay[..., c])
                acc_full = np.maximum(acc_full, on.astype(np.float32))
        acc = np.clip(acc_full, 0.0, 1.0)[..., None]

        a = np.clip(涂色浓度 * acc, 0.0, 1.0)
        out = base * (1 - a) + overlay[None, ...] * a
        return (torch.from_numpy(np.ascontiguousarray(out)).float(),)


NODE_CLASS_MAPPINGS = {
    "MOSSIdRegionMap": MOSSIdRegionMap,
    "MOSSIdRegionPreview": MOSSIdRegionPreview,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSIdRegionMap": "地盘分配器",
    "MOSSIdRegionPreview": "地盘预览图",
}
