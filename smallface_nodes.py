"""小脸精修：把小手够不着的那几张脸，裁出来放大重画，再软边贴回去

为什么非这么做不可（实测，不是推测）：
一张 160 像素宽的脸，在整幅画布上只占大约 10×10 个小格子。格子的数量决定了
这张脸能装多少细节——原地重绘推再多遍，格子还是那 10 个，长不出原来没有的东西。
  实测同一张脸（对参考脸的净相似度，地板 0.625）：
    第二趟基准        −0.142
    原地重绘 0.70     −0.175（越修越差）
    原地重绘 1.00     −0.282（丢脸）
    只加滤镜浓度      −0.390（推崩）
    裁出来放大 4 倍重画再贴回  −0.025 ／ +0.010（两档都翻身）
放大之后同一张脸有 40×40 个格子，滤镜才有地方把"这个人长什么样"写进去；
缩回去时细节是带得回来的，而发色、衣服、构图都不动。

两个节点：
  『小脸裁剪放大』  原图 + ③号地盘 → 每张小脸一张放大裁片（一次出多张，批着算）
  『小脸贴回』      重画好的裁片 → 缩回原尺寸、颜色对齐、软边贴回原图

中间那段完全复用已有节点：裁片地盘直接喂『人物滤镜·分区版』（按编号各挂各的滤镜），
再进 SetLatentNoiseMask（裁片遮罩只管头）和一普通采样器。
"""

import numpy as np
import torch

try:
    import cv2
except Exception:            # 环境缺库时不要静默失效，用中文说清楚
    cv2 = None

DS = 16                      # 底模 1 个图像 token 管 16×16 像素
PACK = 32                    # 裁片边长按 32 的倍数凑：部分 VAE 是 2×2 打包，
                             # 只凑 16 的倍数（例如 1648）会在这里出问题


def _to_np(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _ellipse(H, W, cx, cy, rx, ry):
    """软边椭圆：把整个头（含头发）圈出来，外沿留 45% 的环形带做线性衰减。
    用解析式而不是 cv2.fillEllipse，是因为硬边会在重画区和未重画区之间留台阶。"""
    gy, gx = np.mgrid[0:H, 0:W]
    d = ((gx - cx) / max(1.0, rx)) ** 2 + ((gy - cy) / max(1.0, ry)) ** 2
    m = np.zeros((H, W), np.float32)
    m[d <= 1.0] = 1.0
    edge = (d > 1.0) & (d < 1.45)
    m[edge] = np.clip((1.45 - d[edge]) / 0.45, 0.0, 1.0)
    return m


def _snap16(v):
    """裁片边长：按 32 的倍数凑（不是 16）。"""
    return int(max(PACK, round(v / PACK) * PACK))


class MOSSIdSmallFaceCrop:
    """小脸裁剪放大"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "原图": ("IMAGE", {"tooltip": "要修的那张图（一般接第二趟的成品）。"
                                           "注意：裁的是这张，不是草稿"}),
                "地盘数据": ("REGION_MAP", {"tooltip": "来自『人脸定位地盘』。"
                                                     "每张脸的位置和编号从这里取，"
                                                     "所以修完贴回去以后，编号还是原来那个"}),
                "脸宽上限": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 8,
                                 "tooltip": "多宽算小脸，由你定：脸宽（画面像素）小于这个值的才进"
                                            "放大重画。\n填 0＝沿用『人脸定位地盘』里的"
                                            "『小脸精修上限』，两处填一处即可。\n"
                                            "拿不准就先用一个明显偏小的值试，看说明里谁进、谁不进，再调。"}),
                "裁片尺寸": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 32,
                    "tooltip": "每张裁片统一缩放成多大的正方形（会自动凑成 32 的倍数）。\n"
                               "· 调大：脸在裁片里占的像素更多，细节更有余量，但显存和时间明显上升；\n"
                               "· 调小：更快，但脸的余量变少。\n"
                               "注意：这是整张裁片的尺寸，不是脸的宽度——脸只占其中一部分。\n"
                               "填任意数字都行，节点会自己取最近的 32 的倍数。"}),
                "遮罩形状": (["只脸", "整个头"], {"default": "只脸",
                    "tooltip": "重画哪一块：\n"
                               "· 只脸＝只重画五官那一块，边界落在脸颊/额头/下巴的皮肤上，"
                               "羽化后几乎看不出；头发、脖子、衣服一律不动。\n"
                               "· 整个头＝连头发一起重画，换发型/发色时用，但边界在头发和背景上，"
                               "更容易看出痕迹。"}),
                "留边倍数": ("FLOAT", {"default": 1.15, "min": 0.8, "max": 2.5, "step": 0.05,
                                  "tooltip": "头周围多留多少空间。留太少下巴、头发会被切掉；"
                                             "留太多会把旁边的人也框进来、白吃显存。"}),
                "只修这些编号": ("STRING", {"default": "0",
                    "tooltip": "修哪几个编号的人。\n"
                               "· 填 0＝自动：沿用上一次（第二趟）的编号，只修"
                               "『挂了滤镜』的那几位——大多数情况用这个就行，不用手填；\n"
                               "· 留空＝凡是小于『脸宽上限』的脸都修（没挂滤镜的人重画一遍，"
                               "身份会轻微变差）；\n"
                               "· 填 1,2 这种＝只修这几个编号。"}),
            },
            "optional": {
                "小脸遮罩": ("MASK", {"tooltip": "接『人脸定位地盘』的第 4 个输出，"
                                                 "裁片遮罩就用同一套几何，贴回时边界对得最准。\n"
                                                 "不接也行，节点会按脸框自己算一个椭圆。"}),
                "清单": ("LORA_LIST", {"tooltip": "接最后一个『人物滤镜·分区版』的『清单』。"
                                                  "接上以后『只修这些编号』填 0 就能自动只修挂了滤镜的人，"
                                                  "编号也不用填。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "REGION_MAP", "STRING")
    RETURN_NAMES = ("裁片", "裁片遮罩", "裁片地盘", "说明")
    FUNCTION = "run"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "把每张小脸裁出来放大（一次出多张）；裁片地盘按编号接滤镜，贴回见下一个节点"

    def run(self, 原图, 地盘数据, 脸宽上限, 裁片尺寸, 遮罩形状, 留边倍数, 只修这些编号="0",
            小脸遮罩=None, 清单=None):
        img = 原图[0] if 原图.dim() == 4 else 原图
        a = _to_np(img)
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        H, W = a.shape[:2]
        # 地盘里的脸框是「画布」坐标；原图如果和画布不一样大（比如又做过一次放大），
        # 直接照搬会把遮罩画到脸旁边去。这里统一折算到原图像素。
        _cw, _ch = (地盘数据.get("canvas") or (W, H))
        _sx = W / float(_cw) if _cw else 1.0
        _sy = H / float(_ch) if _ch else 1.0
        boxes = []
        for _b in (地盘数据.get("boxes") or []):
            _bb = dict(_b)
            _bb["x"] = float(_b.get("x") or 0) * _sx
            _bb["y"] = float(_b.get("y") or 0) * _sy
            _bb["w"] = float(_b.get("w") or 0) * _sx
            _bb["h"] = float(_b.get("h") or 0) * _sy
            boxes.append(_bb)
        lim = int(脸宽上限) if int(脸宽上限) > 0 else int(地盘数据.get("小脸上限", 0) or 0)
        spec_txt = str(只修这些编号).strip()
        allow = set()
        auto_pick = spec_txt in ("0", "")
        if auto_pick:
            # 0＝自动：沿用上一次挂了滤镜的那些编号（清单里记着）
            for e in (清单 or []):
                if e.get("lora"):
                    allow.add(int(e.get("owner") or 0))
        else:
            for part in spec_txt.replace("，", ",").split(","):
                part = part.strip()
                if part.isdigit():
                    allow.add(int(part))

        notes = []
        picks = []
        if auto_pick and allow:
            notes.append("『只修这些编号』＝0（自动）→ 只修清单里挂了滤镜的：%s"
                         % "、".join(str(k) for k in sorted(allow)))
            all_owners = sorted({int(b.get("owner") or 0) for b in boxes})
            missing = [k for k in all_owners if k not in allow]
            if missing:
                notes.append("清单里没有 %s 号（地盘里有 %s 号）→ 这几位这一趟不修。如果"
                             "他们本来也该挂滤镜，那就是第 %s 个『人物滤镜·分区版』的"
                             "『清单』输入端没接到前一个的『清单』输出上——"
                             "每人一个滤镜节点时，要像接线一样一路串下来。"
                             % ("、".join(str(k) for k in missing),
                                "、".join(str(k) for k in all_owners),
                                "、".join(str(k) for k in missing)))
        elif auto_pick:
            notes.append("『只修这些编号』＝0（自动），但没接『清单』→ 这张图里所有小脸都会修")

        if lim <= 0:
            notes.append("『脸宽上限』填的是 0，『人脸定位地盘』那边也没设小脸精修上限 → "
                         "判定关着。这一趟原样传下去，不做任何放大重画。")
        else:
            for b in boxes:
                w, h = float(b.get("w") or 0), float(b.get("h") or 0)
                nm = b.get("name") or ("%d号" % int(b.get("owner") or 0))
                if w <= 0 or h <= 0:
                    notes.append("%s：没有脸框尺寸，跳过" % nm)
                    continue
                if w > lim:
                    notes.append("%s：脸宽 %.0f 像素（上限 %d）→ 不算小脸，跳过" % (nm, w, lim))
                    continue
                if allow and int(b.get("owner") or 0) not in allow:
                    notes.append("%s（编号 %d）：不在『只修这些编号』里，跳过"
                                 % (nm, int(b.get("owner") or 0)))
                    continue
                picks.append((b, w, h))
                notes.append("%s：脸宽 %.0f 像素（上限 %d）→ 进放大重画" % (nm, w, lim))

        if not picks:
            # 一张都没进：也要给下游一张合法输入，否则采样器直接报错。
            # 这一张的遮罩是全黑的，重画不出来任何变化，贴回也不会改动任何像素。
            U = _snap16(max(W, H))
            canvas = np.zeros((U, U, 3), np.uint8)
            canvas[:min(U, H), :min(U, W)] = a[:min(U, H), :min(U, W)]
            m = np.zeros((U, U), np.float32)
            region = {"grid_w": U // DS, "grid_h": U // DS, "n_img": (U // DS) ** 2,
                      "ds": DS, "canvas": (U, U),
                      "boxes": [{"owner": 1, "name": "无", "x": 0, "y": 0, "w": U, "h": U}],
                      "weight": torch.zeros((1, (U // DS) ** 2)),
                      "owner": torch.zeros((U // DS, U // DS), dtype=torch.int8),
                      "feather": 0, "mode": "小脸裁剪·空", "按项门控": True,
                      "项归属": [1], "裁片": {"items": [], "canvas": (W, H), "padded": U}} 
            return (torch.from_numpy(canvas).float().div_(255.0)[None],
                    torch.from_numpy(m)[None], region,
                    "\n".join(["小脸精修：" + n for n in notes]
                              + ["没有脸进放大重画，这一趟等于不做事。"])
                    )

        # ── 逐张裁、放大；先算统一尺寸（一个批里的所有图必须一样大）──────────
        plans = []
        S = _snap16(int(裁片尺寸))              # 取最近的 32 的倍数，固定方形画布
        for b, w, h in picks:
            # 先把"头"框出来（脸框再放一点，够到头发和下巴），再把整张缩放到 S
            # 头圈（软边椭圆）横直径 ≈ 1.7 倍脸宽、竖直径 ≈ 1.7 倍脸高（1.15+0.55）。
            # 裁片框必须比它大一圈，否则头圈被裁成方角，贴回去就是一个矩形块。
            # 框要比头圈大得多：头圈直径约 1.7 倍脸高，若框只有 1.7 倍，
            # 头圈会撑满整块裁片 → 贴回去就是一整块方块（实测框内遮罩覆盖 100%）。
            # 给到 2.8 倍，头圈四周留出原样的头发/背景做过渡带（覆盖率降到 55% 左右）。
            # 只脸：框只要比脸大一圈（遮罩直径约 1.16 倍脸宽），框小、放大倍数才够；
            # 整个头：头圈直径约 1.7 倍脸高，框必须给到 2.8 倍，否则头圈被裁成方角。
            k_box = 1.9 if 遮罩形状 == "只脸" else 2.8
            side = max(w, h) * k_box * float(留边倍数)
            k = S / side                        # 整体缩放倍数
            plans.append({"box": b, "w": w, "h": h, "side": side, "k": k, "out": S})
        U = S

        crops, masks, items = [], [], []
        cover_note = []
        mfull = None
        if 小脸遮罩 is not None:
            mfull = _to_np(小脸遮罩)
            if mfull.ndim == 3:
                mfull = mfull[0]
            if mfull.shape[:2] != (H, W):
                if cv2 is not None:
                    mfull = cv2.resize(mfull, (W, H), interpolation=cv2.INTER_LINEAR)
                else:
                    mfull = np.array(torch.nn.functional.interpolate(
                        torch.from_numpy(mfull)[None, None], size=(H, W),
                        mode="bilinear", align_corners=False)[0, 0])

        for pl in plans:
            b = pl["box"]
            cx = float(b["x"]) + pl["w"] / 2.0
            cy = float(b["y"]) + pl["h"] / 2.0
            half = pl["side"] / 2.0
            x0, y0 = int(round(cx - half)), int(round(cy - half))
            x1, y1 = x0 + int(round(pl["side"])), y0 + int(round(pl["side"]))
            # 越界的部分用边缘像素补，不补黑：黑边会被当成真实内容参与计算
            pad_l, pad_t = max(0, -x0), max(0, -y0)
            pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(W, x1), min(H, y1)
            sub = a[sy0:sy1, sx0:sx1]
            if any((pad_l, pad_t, pad_r, pad_b)):
                if cv2 is not None:
                    sub = cv2.copyMakeBorder(sub, pad_t, pad_b, pad_l, pad_r,
                                             cv2.BORDER_REPLICATE)
                else:
                    sub = np.pad(sub, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
            sw, sh = sub.shape[1], sub.shape[0]

            # 遮罩永远按"这个人自己"算：以她的脸框画椭圆，再把别人的脸框整块扣掉。
            # 不能从③号那张合并遮罩里裁——那张把所有人的头放在同一个通道，
            # 裁谁都会把别人的头带进来（结果是每个裁片圈两个人 → 重画时眼睛、手重影）。
            fh = pl["h"]
            if 遮罩形状 == "只脸":
                # 只圈脸框（外放 8%），边界落在皮肤上；头发、脖子、衣服都不进这张遮罩
                # 收紧到脸框内切：边界落在额头/脸颊/下巴的皮肤上，
                # 不要外扩到头发上（外扩那圈正好是头发，最容易看出接缝）
                rx = pl["w"] * 0.48
                ry = fh * 0.50
                own = _ellipse(sh, sw, cx - x0, cy - y0, max(6.0, rx), max(6.0, ry))
            else:
                top, bot = cy - fh * 1.15, cy + fh * 0.55
                own = _ellipse(sh, sw, cx - x0, (top + bot) / 2.0 - y0,
                               max(6.0, pl["w"] * 0.85), (bot - top) / 2.0)
            mx0, my0 = x0, y0                       # 这块裁片（补边后）在原图上的左上角
            for ob in boxes:
                if int(ob.get("owner") or 0) == int(b.get("owner") or 0):
                    continue
                ow, oh = float(ob.get("w") or 0), float(ob.get("h") or 0)
                if ow <= 0 or oh <= 0:
                    continue
                ocx = float(ob["x"]) + ow / 2.0
                ocy = float(ob["y"]) + oh / 2.0
                # 别人的脸框（外扩 15%），换算到这块裁片的坐标后整块清零
                rx0 = int(round(ocx - ow * 0.65 - mx0))
                rx1 = int(round(ocx + ow * 0.65 - mx0))
                ry0 = int(round(ocy - oh * 0.65 - my0))
                ry1 = int(round(ocy + oh * 0.65 - my0))
                rx0, rx1 = max(0, rx0), min(sw, rx1)
                ry0, ry1 = max(0, ry0), min(sh, ry1)
                if rx1 > rx0 and ry1 > ry0:
                    own[ry0:ry1, rx0:rx1] = 0.0
            msub = own

            # 自检：自己脸框盖了多少、别人的脸框盖了多少（后者必须是 0）
            def _cov(rect, mg):
                rx0 = int(round((rect[0] - mx0)))
                rx1 = int(round((rect[1] - mx0)))
                ry0 = int(round((rect[2] - my0)))
                ry1 = int(round((rect[3] - my0)))
                rx0, ry0 = max(0, rx0), max(0, ry0)
                rx1, ry1 = min(sw, rx1), min(sh, ry1)
                if rx1 <= rx0 or ry1 <= ry0:
                    return None
                return float((mg[ry0:ry1, rx0:rx1] > 0.5).mean())

            own_cov = _cov((cx - pl["w"] / 2, cx + pl["w"] / 2,
                            cy - pl["h"] / 2, cy + pl["h"] / 2), msub)
            worst_other, worst_name = 0.0, ""
            for ob in boxes:
                if int(ob.get("owner") or 0) == int(b.get("owner") or 0):
                    continue
                ow2, oh2 = float(ob.get("w") or 0), float(ob.get("h") or 0)
                if ow2 <= 0:
                    continue
                ocx2 = float(ob["x"]) + ow2 / 2.0
                ocy2 = float(ob["y"]) + oh2 / 2.0
                cv = _cov((ocx2 - ow2 / 2, ocx2 + ow2 / 2, ocy2 - oh2 / 2, ocy2 + oh2 / 2), msub)
                if cv is not None and cv > worst_other:
                    worst_other, worst_name = cv, (ob.get("name") or ("%s号" % ob.get("owner")))
            cover_note.append("%s（%s）：自己脸框 %s｜别人脸框 %s%s" % (
                b.get("name") or "无", 遮罩形状,
                "—" if own_cov is None else "%.0f%%" % (100 * own_cov),
                "0%" if worst_other < 0.005 else "%.1f%%" % (100 * worst_other),
                "" if worst_other < 0.005 else "（⚠ 圈到 %s 了）" % worst_name))

            big = cv2.resize(sub, (pl["out"], pl["out"]),
                             interpolation=cv2.INTER_LANCZOS4) if cv2 is not None else \
                np.array(torch.nn.functional.interpolate(
                    torch.from_numpy(sub).permute(2, 0, 1)[None].float(),
                    size=(pl["out"], pl["out"]), mode="bicubic", align_corners=False
                )[0].permute(1, 2, 0).byte().numpy())
            bigm = cv2.resize(msub.astype(np.float32), (pl["out"], pl["out"]),
                              interpolation=cv2.INTER_LINEAR) if cv2 is not None else \
                np.array(torch.nn.functional.interpolate(
                    torch.from_numpy(msub.astype(np.float32))[None, None],
                    size=(pl["out"], pl["out"]), mode="bilinear", align_corners=False)[0, 0])

            # 重画芯比贴回区略小一圈：重画只发生在芯里，贴回时芯外那一圈
            # 用的还是原图内容，所以交界处不会有"新脸压旧脖子"的硬边。
            # FaceDetailer 也是这个思路（先收一点再羽化）。
            if cv2 is not None:
                k = max(3, int(round(min(bigm.shape[:2]) * 0.02)))
                k = k + 1 if k % 2 == 0 else k
                ker = np.ones((k, k), np.uint8)
                er = cv2.erode((bigm > 0.5).astype(np.uint8), ker)
                bigm = cv2.GaussianBlur(er.astype(np.float32), (k, k), 0)

            # 补齐到统一尺寸（右边、下边补边缘，不补黑）
            pl_w, pl_h = pl["out"], pl["out"]
            bigp = np.zeros((U, U, 3), np.uint8)
            bigp[:pl_h, :pl_w] = big
            if U > pl_h or U > pl_w:
                bigp[:pl_h, pl_w:] = big[:, -1:]
                bigp[pl_h:, :] = bigp[pl_h - 1:pl_h, :]
            mp = np.zeros((U, U), np.float32)
            mp[:pl_h, :pl_w] = bigm

            edge_touch = float(min(bigm[0].max(), bigm[-1].max(),
                                   bigm[:, 0].max(), bigm[:, -1].max()))
            if edge_touch > 0.05:
                notes.append("%s：头圈顶到裁片边缘了（边缘遮罩值 %.2f）→ "
                             "会把贴回变成方块边，请把『留边倍数』调大一些"
                             % (b.get("name") or "无", edge_touch))
            crops.append(bigp)
            masks.append(mp)
            items.append({"owner": int(b.get("owner") or 0),
                          "name": b.get("name") or "",
                          "box": [x0, y0, x1, y1],          # 原图坐标，贴回用
                          "size": [pl_w, pl_h],             # 未补齐前的裁片尺寸
                          "k": pl["k"], "face_w": pl["w"]})

        # ── 裁片地盘：每张裁片一块，按项门控，编号照抄原图 ──────────────────
        g = U // DS
        weight = np.stack([cv2.resize(m, (g, g), interpolation=cv2.INTER_AREA) if cv2 is not None
                           else np.array(torch.nn.functional.adaptive_avg_pool2d(
                               torch.from_numpy(m)[None, None], (g, g))[0, 0])
                           for m in masks]).reshape(len(masks), -1)
        region = {"grid_w": g, "grid_h": g, "n_img": g * g, "ds": DS,
                  "canvas": (U, U),
                  "boxes": [{"owner": it["owner"], "name": it["name"],
                             "x": 0, "y": 0, "w": U, "h": U} for it in items],
                  "weight": torch.from_numpy(np.ascontiguousarray(weight, np.float32)),
                  "owner": torch.zeros((g, g), dtype=torch.int8),
                  "feather": 0, "mode": "小脸裁剪",
                  "按项门控": True, "项归属": [it["owner"] for it in items],
                  "裁片": {"items": items, "canvas": (W, H), "padded": U}}

        for it, pl in zip(items, plans):
            notes.append("%s（编号 %d）：裁 %d 像素见方 → 整体缩放 %.2f 倍到 %d×%d"
                         % (it["name"] or "无", it["owner"], int(round(pl["side"])),
                            pl["k"], U, U))
        head = ("小脸精修：共 %d 张小脸进放大重画，全部裁片统一补到 %d×%d（一次批着算）。"
                % (len(items), U, U))
        txt = "\n".join([head] + [n if n.startswith("小脸精修") else "  " + n for n in notes]
                        + (["\n遮罩自检（自己脸框 / 别人脸框）："] + ["  " + c for c in cover_note]
                           if cover_note else [])
                        + ["裁片地盘已经按编号分好：把『人物滤镜·分区版』接在裁片地盘上、"
                           "编号填同一个数字，这个人就只在他自己那张裁片上生效。",
                           "裁片遮罩是按每个人的脸框单独算的（已扣掉别人的脸框），接 SetLatentNoiseMask，保证只重画自己这个头。"])
        return (torch.from_numpy(np.stack(crops)).float().div_(255.0),
                torch.from_numpy(np.stack(masks)),
                region, txt)


class MOSSIdSmallFacePaste:
    """小脸贴回"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "原图": ("IMAGE", {"tooltip": "要贴回的那张图；通常和『小脸裁剪放大』的"
                                           "『原图』是同一张"}),
                "重画裁片": ("IMAGE", {"tooltip": "放大裁片重画后的结果，接采样器后面的解码"}),
                "裁片地盘": ("REGION_MAP", {"tooltip": "接『小脸裁剪放大』的『裁片地盘』，"
                                                     "里面记着每张裁片原来是从哪儿裁的"}),
                "原图遮罩": ("MASK", {"tooltip": "接『人脸定位地盘』的第 4 个输出（小脸遮罩）。"
                                              "贴回只发生在这张遮罩里，圈外一个像素都不动"}),
                "颜色匹配": (["开", "关"], {"default": "开",
                    "tooltip": "缩回来的脸和周围是从不同分辨率来的，直接贴容易有一圈色差。\n"
                               "开＝把新脸的颜色亮度对齐到原图那块再贴，接口更自然；\n"
                               "关＝原样贴，想保留重画出来的原始色调时用。"}),
                "边缘羽化": ("INT", {"default": 0, "min": 0, "max": 128, "step": 2,
                                 "tooltip": "贴回时交界处抹软多少像素。\n"
                                            "填 0＝按画面大小自动；总觉得脖子那儿能看出接缝，"
                                            "就先小步往上加，盖住接缝就停。"}),
                "清晰度补偿": (["开", "关"], {"default": "开",
                    "tooltip": "小脸要「放大再缩回」，这一来一回会磨掉一部分高频细节，"
                               "重画也不会把细节长回来。\n"
                               "开＝贴回后在贴回区域内做一次轻度锐化，把这层模糊补回来；\n"
                               "关＝原样贴，想保留模型出来的原始质感时用。"}),
            },
            "optional": {
                "裁片遮罩": ("MASK", {"tooltip": "接『小脸裁剪放大』的『裁片遮罩』。\n"
                                                      "可选。接上以后，贴回用的是「重画过的那一块」，"
                                                      "而不是③号那张大椭圆——"
                                                      "大椭圆的边缘那一圈是「放大再缩回」的软图，"
                                                      "贴回去会出现一圈发虚的痕迹。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("成品", "说明")
    FUNCTION = "run"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "把放大重画好的小脸缩回原尺寸、颜色对齐、软边贴回原图"

    def run(self, 原图, 重画裁片, 裁片地盘, 原图遮罩, 颜色匹配, 边缘羽化,
            清晰度补偿="开", 裁片遮罩=None):
        base = 原图[0] if 原图.dim() == 4 else 原图
        a = _to_np(base)
        if a.dtype != np.uint8:
            a = np.clip(a * 255.0, 0, 255).astype(np.uint8)
        H, W = a.shape[:2]
        out = a.astype(np.float32)
        mk = _to_np(原图遮罩)
        if mk.ndim == 3:
            mk = mk[0]
        mk = mk.astype(np.float32)
        if mk.shape[:2] != (H, W) and cv2 is not None:
            mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_LINEAR)
        mk = np.clip(mk, 0.0, 1.0)

        cmask = _to_np(裁片遮罩) if 裁片遮罩 is not None else None
        if cmask is not None and cmask.ndim == 2:
            cmask = cmask[None]
        crops = _to_np(重画裁片)
        if crops.ndim == 3:
            crops = crops[None]
        if crops.dtype != np.uint8:
            crops = np.clip(crops * 255.0, 0, 255).astype(np.uint8)

        info = 裁片地盘.get("裁片") or {}
        items = list(info.get("items") or [])
        if not items:
            return (原图, "裁片地盘里没有小脸（判定关着，或一张都没进）→ 原图原样输出，"
                          "这一趟不做事。")

        f = int(边缘羽化)
        auto_f = f <= 0
        if auto_f:
            f = 0                      # 先占位，进循环后按每块的实际大小算
        notes = []
        # 多张裁片会互相重叠（两个人挨着时）。后一张直接盖前一张会在重叠处留下
        # 二次混合的印子——这就是"双人接缝、单人没事"的原因。
        # 改成先把所有张按权重累加，再一次性合成：
        #   结果 = 原图 × (1 - min(1,Σw)) + Σ(w × 新内容) / max(1, Σw)
        acc_w = np.zeros((H, W), np.float32)
        acc_c = np.zeros((H, W, 3), np.float32)
        for i, it in enumerate(items[:crops.shape[0]]):
            x0, y0, x1, y1 = [int(v) for v in it["box"]]
            bw, bh = max(1, x1 - x0), max(1, y1 - y0)
            plw, plh = [int(v) for v in it["size"]]
            c = crops[i][:plh, :plw]
            # 羽化按"这一块"的短边算（约 2.5%），大块多抹、小块少抹：
            # 固定 12 像素对 200 的小块太多、对 900 的大块又太少，交界就会显得硬。
            fi = f if not auto_f else int(min(48, max(6, round(min(bw, bh) * 0.025))))
            small = cv2.resize(c, (bw, bh), interpolation=cv2.INTER_AREA) if cv2 is not None \
                else np.array(torch.nn.functional.interpolate(
                    torch.from_numpy(c).permute(2, 0, 1)[None].float(),
                    size=(bh, bw), mode="bilinear", align_corners=False
                )[0].permute(1, 2, 0).byte().numpy())
            small = small.astype(np.float32)

            # 只取这块的遮罩；越界部分裁掉
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(W, x1), min(H, y1)
            if cmask is not None and i < cmask.shape[0]:
                # 用裁片自己的遮罩：它只圈"真正重画过"的那一块（还缩了 2%），
                # 所以贴回去时，外面那一圈用的还是原图像素，不会有一圈发虚的边。
                src = cmask[i][:plh, :plw]
                if cv2 is not None:
                    src = cv2.resize(src, (bw, bh), interpolation=cv2.INTER_LINEAR)
                else:
                    src = np.array(torch.nn.functional.interpolate(
                        torch.from_numpy(src)[None, None], size=(bh, bw),
                        mode="bilinear", align_corners=False)[0, 0])
                m = src[sy0 - y0:sy0 - y0 + (sy1 - sy0), sx0 - x0:sx0 - x0 + (sx1 - sx0)][..., None]
            else:
                m = mk[sy0:sy1, sx0:sx1][..., None]
            if m.size == 0:
                continue
            sm = small[sy0 - y0:sy0 - y0 + (sy1 - sy0), sx0 - x0:sx0 - x0 + (sx1 - sx0)]
            if sm.shape[:2] != m.shape[:2]:
                continue

            if 颜色匹配 == "开" and cv2 is not None:
                sel = m[..., 0] > 0.5
                if sel.sum() > 64:
                    src = cv2.cvtColor(sm.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
                    tgt = cv2.cvtColor(out[sy0:sy1, sx0:sx1].astype(np.uint8),
                                       cv2.COLOR_RGB2LAB).astype(np.float32)
                    for ch in range(3):
                        s_c = src[..., ch][sel]
                        t_c = tgt[..., ch][sel]
                        sd = s_c.std() or 1.0
                        src[..., ch] = (src[..., ch] - s_c.mean()) / sd * (t_c.std() or 1.0) \
                                       + t_c.mean()
                    sm = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8),
                                      cv2.COLOR_LAB2RGB).astype(np.float32)

            # 羽化：把交界抹软，避免"贴图感"。
            # 抹软只允许发生在原遮罩自身覆盖的范围内——否则高斯会把权重溢出圈外，
            # "圈外一个像素都不动"就不成立了。
            if cv2 is not None and fi > 0:
                mm = cv2.GaussianBlur(m[..., 0], (2 * fi + 1, 2 * fi + 1), 0)[..., None]
                mm = mm * (m > 0)
            else:
                mm = m
            acc_w[sy0:sy1, sx0:sx1] += mm[..., 0]
            acc_c[sy0:sy1, sx0:sx1] += sm * mm
            notes.append("%s（编号 %d）：贴回 x %d~%d y %d~%d"
                         % (it.get("name") or "无", it.get("owner") or 0, x0, x1, y0, y1))

        if acc_w.max() > 0:
            wsum = np.maximum(acc_w, 1e-6)[..., None]
            newc = acc_c / wsum
            cover = np.clip(acc_w, 0.0, 1.0)[..., None]
            out = out * (1 - cover) + newc * cover
            # 放大再缩回会把高频磨掉（实测脸中心清晰度 178 → 119），模型重画也补不回来
            # （0.55/0.75 都是 104~109，2048 还是 109）。这里只在贴回区域内补一次轻度锐化。
            if 清晰度补偿 == "开" and cv2 is not None:
                soft = cv2.GaussianBlur(out, (0, 0), 1.0)
                back = np.clip(out + 0.30 * (out - soft), 0, 255)
                out = out * (1 - cover) + back * cover

        txt = "\n".join(["小脸贴回：共贴回 %d 张，羽化%s，颜色匹配%s，清晰度补偿%s。"
                         % (len(notes), "按块自动" if auto_f else ("%d 像素" % f), 颜色匹配, 清晰度补偿)]
                        + ["  " + n for n in notes]
                        + ["羽化以外的区域逐像素保持原样，构图不会被挪动。"])
        return (torch.from_numpy(np.clip(out, 0, 255).astype(np.uint8)).float().div_(255.0)[None],
                txt)


NODE_CLASS_MAPPINGS = {"MOSSIdSmallFaceCrop": MOSSIdSmallFaceCrop,
                       "MOSSIdSmallFacePaste": MOSSIdSmallFacePaste}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSIdSmallFaceCrop": "小脸裁剪放大",
                              "MOSSIdSmallFacePaste": "小脸贴回"}
