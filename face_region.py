"""人脸定位地盘：按画面里真实的人脸位置，把画布切成"每人一块地盘"

输入一张草稿图，自动完成三件事：
  1) 找出草稿图里每张脸的位置，按位置顺序编号 1、2、3……
  2) 按『遮罩来源』决定每个人能管到哪些像素：不接＝就近＋半径；
     外部接入＝用你接进来的遮罩；自动逐人切＝本节点调用 SAM 各切一块
  3) 把结果换算成模型使用的小格子归属表，边界按『边界抹软像素』过渡

草稿里有人被挡住、低头、埋住而检测不到时不报错：
能对上的人用真实脸位置，对不上的人退回『备用方框』里的位置，并在说明里提醒。

输出的数据结构和「地盘分配器」完全一样，所以「人物滤镜·分区版」直接接上就能用。
"""

import json

import numpy as np
import torch

try:
    from comfy.utils import common_upscale  # 只用来确认在 ComfyUI 环境里，不参与计算
except Exception:
    pass

try:
    from insightface.app import FaceAnalysis
except Exception as _e:
    FaceAnalysis = None
    _IMPORT_ERROR = str(_e)
else:
    _IMPORT_ERROR = ""

_SC = None


def _self_check_mod():
    """按需加载环境检查模块（ComfyUI 把插件当包导入，绝对导入会失败，所以两种都试）"""
    global _SC
    if _SC is None:
        try:
            from . import self_check as _sc
        except Exception:
            import importlib.util
            import os as _os
            spec = importlib.util.spec_from_file_location(
                "self_check", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "self_check.py"))
            _sc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_sc)
        _SC = _sc
    return _SC


DS = 16
_PM = None


def _person_masks_mod():
    """按需加载同插件里的 SAM 模块（ComfyUI 把插件当包导入，绝对导入会失败）"""
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

_APP = None


def _insightface_root():
    """人脸模型放哪：交给 ComfyUI 登记的 models 目录，找不到再按文件位置推算。
    不能写死成本机路径——那样装到别人机器上就直接找不到，库的自动下载也无从下手。"""
    try:
        return _self_check_mod().INSIGHTFACE_ROOT
    except Exception:
        import os as _os
        return _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "models", "insightface")


def _app():
    global _APP
    if FaceAnalysis is None:
        raise RuntimeError(
            "这台机器的 ComfyUI Python 里没有 insightface，用不了人脸定位。\n"
            "省事的一步：ComfyUI-Manager 里对本插件点 Reinstall（或双击插件文件夹里的 install.bat），"
            "然后重启 ComfyUI。\n退路（手动）：pip install insightface，装完必须重启 ComfyUI。"
            + ("\n（导入时的原始报错：%s）" % _IMPORT_ERROR if _IMPORT_ERROR else ""))
    if _APP is None:
        # 模型不在的时候，insightface 库会自己从官方发布页下载 antelopev2 并解压，
        # 这里只要把 root 指对，不需要我们替它下（重复下反而多占 300MB）
        _APP = FaceAnalysis(name="antelopev2", root=_insightface_root(),
                            allowed_modules=["detection", "recognition"],
                            providers=["CPUExecutionProvider"])
        _APP.prepare(ctx_id=0, det_size=(1024, 1024))
    return _APP


def _to_bgr(image_tensor):
    arr = image_tensor.detach().cpu().float().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr[..., ::-1]


DIGITS = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00110", "01000", "10000", "11111"),
    "3": ("11111", "00010", "00100", "00010", "00001", "10001", "01110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
}


def _rect(img, x0, y0, x1, y1, rgb, t=6):
    h, w = img.shape[0], img.shape[1]
    x0, x1 = max(0, int(x0)), min(w - 1, int(x1))
    y0, y1 = max(0, int(y0)), min(h - 1, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    img[y0:y0 + t, x0:x1] = rgb
    img[max(y0, y1 - t):y1, x0:x1] = rgb
    img[y0:y1, x0:x0 + t] = rgb
    img[y0:y1, max(x0, x1 - t):x1] = rgb


def _put_text(img, text, x, y, rgb, s=10, bg=None):
    """用 5x7 点阵把数字画成大字（不依赖任何字体文件），bg 给一块深色底衬保证看得清"""
    h, w = img.shape[0], img.shape[1]
    cw, ch = 5 * s + s, 7 * s
    if bg is not None:
        y0, x0 = max(0, int(y - s)), max(0, int(x - s))
        y1, x1 = min(h, y0 + ch + 2 * s), min(w, x0 + cw * len(str(text)) + s)
        img[y0:y1, x0:x1] = bg
    for k, ch_ in enumerate(str(text)):
        g = DIGITS.get(ch_)
        if g is None:
            continue
        for r, row in enumerate(g):
            for c, bit in enumerate(row):
                if bit != "1":
                    continue
                yy, xx = int(y + r * s), int(x + k * cw + c * s)
                img[max(0, yy):min(h, yy + s), max(0, xx):min(w, xx + s)] = rgb


def _faces(bgr):
    app = _app()
    out = []
    for i in range(bgr.shape[0]):
        for f in app.get(bgr[i]):
            box = f.bbox.astype(np.float32)
            e = f.embedding.astype(np.float32)
            out.append({"cx": float((box[0] + box[2]) / 2), "cy": float((box[1] + box[3]) / 2),
                        "w": float(box[2] - box[0]), "h": float(box[3] - box[1]),
                        "emb": e / max(1e-8, float(np.linalg.norm(e))),
                        "conf": float(f.det_score)})
    return out


def _greedy_assign(ref_embs, faces):
    """每个人挑一张最像的、还没被别人占用的脸。返回 [(脸下标或None, 相似度), ...]"""
    used = set()
    pairs = []
    for ri, r in enumerate(ref_embs):
        best, bi = -9.0, None
        for fi, f in enumerate(faces):
            if fi in used:
                continue
            s = float(r @ f["emb"])
            if s > best:
                best, bi = s, fi
        if bi is not None and best > 0.0:
            used.add(bi)
            pairs.append((bi, best))
        else:
            pairs.append((None, 0.0))
    return pairs


def _parse_backup(text):
    if not text or not str(text).strip():
        return None
    try:
        data = json.loads(text)
    except Exception:
        return None
    out = {}
    for i, d in enumerate(data if isinstance(data, list) else []):
        try:
            out[int(d.get("owner", i + 1))] = (int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]))
        except Exception:
            continue
    return out or None


class MOSSIdFaceRegionMap:
    """人脸定位地盘：草稿图 + 参考脸 → 自动分地盘"""

    @staticmethod
    def _env_silent(要用自动切):
        """开工前静默自查：齐了半个字都不说；缺权重自己下回来；补不动才说一句。

        刻意不做成节点、也不给开关——装依赖是插件自己的事，不该让使用者先去
        找一个"体检节点"跑一遍才知道少了什么。
        """
        try:
            sc = _self_check_mod()
            need, opt, weights = sc.missing_items()
        except Exception as exc:
            return ["【环境】自查没能运行（%s），本次按原流程继续，不影响出图" % type(exc).__name__]
        if not 要用自动切:
            weights = []                      # 不自动切就不用占 309MB 去下 SAM2
        else:
            weights = [w for w in weights if "SAM2" in w[0]]
        if need:
            raise RuntimeError("缺少必需项，无法检脸分地盘：%s\n%s" % (
                "、".join(need), sc.one_click_fix()))
        out = []
        if weights:
            done, bad = sc.fetch_weights(weights)
            out += ["【环境】" + s for s in done]
            if bad:
                out += ["【环境】%s\n【环境】%s" % (s, sc.one_click_fix()) for s in bad]
        if 要用自动切 and "sam2" in opt:
            out.append("【环境】这台机器还没有 sam2 这个库，『自动逐人切』暂时用不了，"
                       "可以先改成『外部接入』或『不接遮罩』照常跑。%s" % sc.one_click_fix())
        return out

    @classmethod
    def INPUT_TYPES(cls):
        pm = _person_masks_mod()
        return {
            "required": {
                "草稿图": ("IMAGE", {"tooltip": "用来定位人脸的图。通常是第一趟（不带人物滤镜）出的草稿，"
                                                 "只取脸的位置，不会修改这张图"}),
                "画布宽": ("INT", {"default": 1440, "min": 64, "max": 8192, "step": DS, "tooltip": '画面宽度（像素）。要和实际出图用的画布尺寸一致，否则格子会错位；建议取 16 的倍数。'}),
                "画布高": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": DS, "tooltip": '画面高度（像素）。要和实际出图用的画布尺寸一致；不是 16 的倍数时，最后几行会不属于任何人。'}),
                "归属半径": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8,
                                    "tooltip": "离一张脸多远之内还算这个人。\n"
                                               "0＝自动：接了每人一块遮罩时不参与；接整幅剪影时不设上限；"
                                               "什么都没接时按脸与脸的距离自动算。\n"
                                               "手填时要注意：数值小于人物之间的距离，会把身体切成"
                                               "「谁都不是」，那一块拿不到任何滤镜。"}),
                "边界抹软像素": ("INT", {"default": 24, "min": 0, "max": 512, "step": 1,
                                      "tooltip": "交界处过渡多宽。0＝硬切；越大过渡越平缓，"
                                                 "两个人交界那一圈会按比例分摊"}),
                "角色名字": ("STRING", {"default": "",
                                    "tooltip": "留空＝自动：画面里检到几张脸就算几个人，"
                                               "按位置自动编号 1、2、3……\n"
                                               "也可以只填前面几个名字，后面的人照样编号；"
                                               "没接对应滤镜的人保持原样，不会被换脸。\n"
                                               "人数取「填的名字个数」与「检到的脸数」中较多的那个。"}),
                "谁是谁": (["按位置排（左到右）", "按位置排（上到下）"],
                          {"default": "按位置排（左到右）",
                           "tooltip": "编号顺序。左到右＝画面最左边的脸是 1 号；"
                                      "上下错落的构图（台阶、楼梯、站与蹲）选上到下。\n"
                                      "滤镜节点里填的『人物编号』就按这里的序号来。"}),
                "遮罩来源": (["不接遮罩（只按脸就近＋半径）", "外部接入（下面『遮罩』口）",
                            "自动逐人切（本节点内置调用）"],
                           {"default": "自动逐人切（本节点内置调用）",
                            "tooltip": "决定每个人能管到哪些像素：\n"
                                       "· 不接＝只按脸的位置就近划分；\n"
                                       "· 外部接入＝用你自己接进来的遮罩（见『遮罩』口），"
                                       "别的 SAM 插件出的遮罩也走这一档；\n"
                                       "· 自动逐人切＝本节点自己调用下面选的 SAM 模型，"
                                       "把每个人各切一块，不需要再加别的节点。"}),
                "遮罩放大像素": ("INT", {"default": 0, "min": 0, "max": 192, "step": 4,
                                      "tooltip": "把每块遮罩向外胀多少像素，用来补回被切掉的发梢、衣角、肩膀。\n"
                                                 "0＝最贴身体，换脸最稳；24~48＝覆盖更全，换脸强度略降；"
                                                 "96 以上容易和相邻的人连成一片。"}),
                "SAM模型": (pm.sam_labels(), {"default": pm.sam_default_label(),
                                           "tooltip": "『自动逐人切』用哪个 SAM 权重，只在上一项选"
                                                      "『自动逐人切』时生效。\n"
                                                      "一行三段：能不能直接用 ｜ 放在哪个文件夹 ｜ 多大。"
                                                      "清单是扫描本机 models 里的 SAM2 / SAM3 / SAM 得到的。\n"
                                                      "默认的 SAM2 权重本机没有时，第一次用会自动从官方地址"
                                                      "下载（约 309MB，只一次）；下载失败会在说明里说一声。\n"
                                                      "选到标『需外接节点』『格式不支持』的（例如手头的 SAM3），"
                                                      "本节点不会假装支持，说明里会给出确切的两步接法。"}),
            },
            "optional": {
                "遮罩": ("MASK", {
                    "tooltip": "一个口子，两种都能接，节点按张数自动判断：\n"
                              "· 1 张＝整幅人像剪影，只区分人和背景，背景不发滤镜，"
                              "人和人之间仍按离谁的脸近划分；\n"
                              "· 多张＝一人一块，每个人只能在自己那块里生效，"
                              "块里没有脸的（例如从画面外伸进来的手）不发任何滤镜。"}),
                "参考脸": ("IMAGE", {
                    "tooltip": "可不接。接了只用于在说明里给出「这张脸和参考像不像」的参考分，"
                              "不参与划分地盘，也不负责换脸（换脸靠滤镜文件本身）"}),
                "备用方框": ("STRING", {
                    "tooltip": "某张脸被挡住、检不到时的兜底位置（一般接构图节点输出的方框）。"
                              "没有它时，检不到的人会按人数均分并给出提示"}),
            },
        }

    RETURN_TYPES = ("REGION_MAP", "STRING", "IMAGE")
    RETURN_NAMES = ("地盘数据", "指派说明", "地盘可视化")
    FUNCTION = "build"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "按草稿图里真实人脸位置分地盘；可接遮罩只算人身上的像素；脸被挡住时自动退回备用方框"

    def build(self, 草稿图, 画布宽, 画布高, 归属半径, 边界抹软像素, 角色名字, 谁是谁,
               遮罩来源, 遮罩放大像素, SAM模型,
               遮罩=None, 参考脸=None, 备用方框=None):
        pm = _person_masks_mod()
        auto_cut = not str(遮罩来源).startswith(("不接", "外部"))
        env_lines = self._env_silent(auto_cut)
        sam_row = pm.sam_row(SAM模型)
        sam_note = "SAM 模型：%s ｜ %s ｜ 本机路径：%s" % (
            sam_row["相对路径"], sam_row["结论"], sam_row["绝对路径"])
        names = [s.strip() for s in str(角色名字).replace("，", ",").split(",") if s.strip()]
        draft = _to_bgr(草稿图)
        refs = _to_bgr(参考脸) if 参考脸 is not None else None
        dfaces = _faces(draft)
        if not dfaces:
            raise RuntimeError("草稿图里一张脸都没检到。要么这张图里确实没有正脸，"
                               "要么换一张脸看得清的草稿（第一趟不带人物滤镜出的那张）。")

        # 人数＝「角色名字填的个数」和「草稿里检到的脸数」里取多的那个：
        # 名字留空＝全自动（检到几张脸就几个人）；只填两个名字但画面里有 8 个人，
        # 也照样给 8 张脸编号，只是 3 号以后自动叫「3号、4号…」，不接滤镜就保持原生脸。
        n_people = max(len(names), len(dfaces), refs.shape[0] if refs is not None else 0)
        if n_people > 12:
            raise RuntimeError(f"最多支持 12 个人，这张草稿里检到 {len(dfaces)} 张脸。")
        for i in range(len(names), n_people):
            names.append(f"{i + 1}号")
        given = len([s for s in str(角色名字).replace("，", ",").split(",") if s.strip()])
        auto_note = ("人数全自动：检到几张脸就几个人（共 %d 个）" % n_people if given == 0
                     else "填了 %d 个名字，草稿检到 %d 张脸 → 共 %d 人，%d 号以后自动编号" % (
                         given, len(dfaces), n_people, given + 1))
        backup = _parse_backup(备用方框)

        # ── 遮罩统一入口：一个口子，两种来源都能接 ──────────────────────────
        # 1 张（整幅人像剪影）＝只区分人和背景；N 张（每人一块）＝每人只能在自己块里。
        # 按张数自动判断，不用你选。也可以让本节点自己去调 SAM 切，省掉一个中间节点。
        mask_raw = 遮罩
        mask_src = "外部接入"
        sam_guide = ""
        if auto_cut:
            usable, _r = pm.sam_is_builtin(SAM模型)
            if not usable:
                # 不假装支持：选中了本插件调不动的模型（例如 SAM3），就把两步接法写清楚
                mask_src = "⚠ 自动逐人切没开：选的是「%s」（%s）" % (
                    sam_row["相对路径"], sam_row["结论"])
                sam_guide = (
                    "想用这个模型，两步："
                    "① 用带它的节点出遮罩（本机是 ComfyUI-RMBG 的『SAM3Segment』，"
                    "模型选同名那个，提示词写 person / 人名，output_mode 选 Separate 才是一人一块），"
                    "把它的 MASK 输出接到本节点的『遮罩』口；"
                    "② 把本节点的『遮罩来源』改成『外部接入（下面『遮罩』口）』。"
                    "这样本节点不再自己调用 SAM，只用你接进来的块，其余参数不用动。")
            else:
                try:
                    rgb = np.clip(草稿图[0].float().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
                    kf = (lambda d: d["cx"]) if "左" in 谁是谁 else (lambda d: d["cy"])
                    fs0 = sorted(dfaces, key=kf)[:max(1, n_people)]
                    got = pm.predict_person_masks(rgb, fs0, SAM模型, "cuda")
                    if got:
                        mask_raw = torch.from_numpy(np.stack(got)).float()
                        mask_src = "自动逐人切（%s，%d 块）" % (sam_row["相对路径"], len(got))
                except Exception as exc:
                    mask_src = "⚠ 自动逐人切失败（%s）%s" % (
                        type(exc).__name__, "，改用外部遮罩" if mask_raw is not None else "，这趟不遮罩")
        expand = int(遮罩放大像素)
        mask_tensor, per_person = None, False
        if mask_raw is not None:
            mm = mask_raw.detach().cpu().float().numpy()
            if mm.ndim == 2:
                mm = mm[None]
            if expand > 0:
                import cv2
                kk = 2 * max(1, int(round(expand / 2.0))) + 1
                ker = np.ones((kk, kk), np.uint8)
                mm = np.stack([cv2.dilate(x, ker) for x in mm]).astype(np.float32)
            per_person = mm.shape[0] > 1
            mask_tensor = torch.from_numpy(np.ascontiguousarray(mm))

        ref_embs, ref_warn = [], []
        if refs is not None:
            for i in range(min(n_people, refs.shape[0])):
                fs = _faces(refs[i:i + 1])
                if not fs:
                    ref_embs.append(None)
                    ref_warn.append(f"第 {i + 1} 张参考脸里检不到正脸，这个人的体检分没有（不影响分地盘）")
                else:
                    ref_embs.append(max(fs, key=lambda d: d["w"] * d["h"])["emb"])

        # 归属半径填 0 ＝ 自动：接了整幅人像剪影就不设上限（背景由剪影负责），
        # 每人一块时半径根本不参与（边界由遮罩决定），没接遮罩才按脸距与脸宽算。
        if int(归属半径) <= 0:
            ws = [f["w"] for f in dfaces]
            cs = [(f["cx"], f["cy"]) for f in dfaces]
            dmin = 1e9
            for a in range(len(cs)):
                for b in range(a + 1, len(cs)):
                    dmin = min(dmin, float(np.hypot(cs[a][0] - cs[b][0], cs[a][1] - cs[b][1])))
            if not cs or dmin > 1e8:
                dmin = float(min(画布宽, 画布高))
            if mask_tensor is not None and not per_person:
                R = 4096.0
                radius_note = "归属半径＝自动 4096（整幅剪影负责背景，人之间不设上限）"
            elif mask_tensor is not None and per_person:
                R = 4096.0
                radius_note = "归属半径＝不参与（每人一块遮罩就是硬边界）"
            else:
                R = min(4096.0, max(1.5 * dmin, 5.0 * float(np.median(ws))))
                radius_note = "归属半径＝自动 %d 像素（没接遮罩＝最近两脸距离1.5倍 与 5倍脸宽 取大）" % R
        else:
            R = float(归属半径)
            radius_note = ""
        # 编号只按位置排（相似度那条路已删：草稿脸是底模画的，跟参考脸不像，会把 1、2 号对调）
        keyf = (lambda d: d["cx"]) if "左" in 谁是谁 else (lambda d: d["cy"])
        order = sorted(range(len(dfaces)), key=lambda i: keyf(dfaces[i]))
        picks = []
        for ri in range(n_people):
            if ri < len(order):
                fi = order[ri]
                e = ref_embs[ri] if ri < len(ref_embs) else None
                picks.append((fi, float(e @ dfaces[fi]["emb"]) if e is not None else None))
            else:
                picks.append((None, None))

        gw = max(1, int(画布宽) // DS)
        gh = max(1, int(画布高) // DS)
        n_img = gw * gh
        col_c = (np.arange(gw, dtype=np.float32) + 0.5) * DS
        row_c = (np.arange(gh, dtype=np.float32) + 0.5) * DS
        CX, CY = np.meshgrid(col_c, row_c)
        px, py = CX.ravel(), CY.ravel()

        R = float(R)
        chosen, scores, notes = [], [], []
        for i, (fi, sim) in enumerate(picks):
            nm = names[i] if i < len(names) else f"{i + 1}号"
            if fi is not None:
                f = dict(dfaces[fi])
                f["来源"] = ("真实检测到（体检分 %.3f，置信度 %.2f）" % (sim, f["conf"])
                             if sim is not None else
                             "真实检测到（没接参考脸，无体检分；置信度 %.2f）" % f["conf"])
            else:
                b = (backup or {}).get(i + 1)
                if b is None:
                    seg = 画布宽 // max(1, n_people)
                    b = (i * seg + int(seg * 0.1), 120, int(seg * 0.8), 画布高 - 240)
                    f = {"cx": b[0] + b[2] / 2.0, "cy": b[1] + b[3] / 2.0,
                         "w": b[2], "h": b[3], "conf": 0.0}
                    f["来源"] = "⚠ 草稿里没检测到她的脸，也没给备用方框，只能按人数均分"
                else:
                    f = {"cx": b[0] + b[2] / 2.0, "cy": b[1] + b[3] / 2.0,
                         "w": b[2], "h": b[3], "conf": 0.0}
                    f["来源"] = "⚠ 草稿里没检测到她的脸（被挡住或低头），已退回第一趟的方框"
                notes.append(f"{nm}：{f['来源']}")
            chosen.append(f)
            scores.append(R - np.sqrt((px - f["cx"]) ** 2 + (py - f["cy"]) ** 2))
        scores = np.stack(scores, axis=0).astype(np.float32)

        # ── 每人一块遮罩：有它就以此为准 ─────────────────────────────────────
        # 规则：① 每张脸落在哪块遮罩里，那块就归这个人，他的地盘只能在自己遮罩内；
        #       ② 没有任何脸的遮罩（主视角观察者的手、背影）一律不发滤镜；
        #       ③ 两个人抢同一块遮罩时，这块按离谁的脸近切开。
        body_note = ""

        if mask_tensor is not None and per_person:
            try:
                bm = mask_tensor.detach().cpu().float().numpy()
                if bm.ndim == 2:
                    bm = bm[None]
                ys = np.linspace(0, bm.shape[1], gh + 1).astype(int)
                xs = np.linspace(0, bm.shape[2], gw + 1).astype(int)
                grid = np.stack([[[float(bm[j, a:b, c:d].mean()) for c, d in zip(xs[:-1], xs[1:])]
                                  for a, b in zip(ys[:-1], ys[1:])] for j in range(bm.shape[0])])
                grid = grid.reshape(bm.shape[0], n_img)
                # 每块遮罩的质心（像素），给"脸所在格没被盖住"时兜底用
                cen = []
                for j in range(grid.shape[0]):
                    w = np.clip(grid[j], 0.0, 1.0)
                    s = float(w.sum())
                    if s <= 1e-6:
                        cen.append((None, None))
                    else:
                        cen.append((float((w * px).sum() / s), float((w * py).sum() / s)))
                pick, used = [], set()
                for i, f in enumerate(chosen):
                    gy = int(min(gh - 1, max(0, f["cy"] // DS)))
                    gx = int(min(gw - 1, max(0, f["cx"] // DS)))
                    gi = gy * gw + gx
                    j = int(grid[:, gi].argmax())
                    val = float(grid[j, gi])
                    if val <= 0.2:
                        # 脸心正好落在遮罩缝里：改挑"质心离这张脸最近"的那块（限 2.5 倍脸宽内）
                        lim = 2.5 * max(24.0, float(f["w"]))
                        cand = [(np.hypot(cen[jj][0] - f["cx"], cen[jj][1] - f["cy"]), jj)
                                for jj in range(grid.shape[0])
                                if cen[jj][0] is not None and
                                np.hypot(cen[jj][0] - f["cx"], cen[jj][1] - f["cy"]) <= lim]
                        if cand:
                            j = min(cand)[1]
                            val = 0.21
                            notes.append("%s：脸心没被任何遮罩盖住，已退回质心最近的那块（%d 号遮罩）" % (
                                names[i], j + 1))
                    pick.append(j if val > 0.2 else -1)
                    if val > 0.2:
                        used.add(j)
                    else:
                        notes.append("%s：这张脸附近一块遮罩都没有（自动切把这个人漏了）" % names[i])
                dropped = [j for j in range(grid.shape[0]) if j not in used]
                own = np.zeros((len(chosen), n_img), dtype=np.float32)
                for i, j in enumerate(pick):
                    if j < 0:
                        continue
                    same = [k for k, jj in enumerate(pick) if jj == j and k != i]
                    if same:
                        di = np.hypot(px - chosen[i]["cx"], py - chosen[i]["cy"])
                        dj = np.stack([np.hypot(px - chosen[k]["cx"], py - chosen[k]["cy"]) for k in same])
                        keep = grid[j] * (di <= dj.min(axis=0))
                    else:
                        keep = grid[j]
                    own[i] = np.maximum(own[i], keep)
                # 遮罩是 0/1 的硬边界，跨度太小会被 24 像素的软过渡抹成一半权重，
                # 所以这里把"在块内/在块外"拉开成 ±120（相当于 240 像素的距离差），
                # 保证块内的格子基本全额拿到自己的滤镜。
                scores = (np.clip(own, 0.0, 1.0) - 0.5) * 240.0
                darea = int((np.stack([grid[j] for j in dropped]) > 0.5).sum()) if dropped else 0
                body_note = ("每人一块遮罩已生效：%d 块遮罩，%d 块有主人，%d 块没有脸（%d 格）不发任何滤镜"
                             % (grid.shape[0], len(used), len(dropped), darea))
            except Exception as exc:
                body_note = "⚠ 每人一块遮罩没用上（%s），这一趟退回「按脸就近＋半径」" % type(exc).__name__

        pool = np.concatenate([scores, np.zeros((1, n_img), dtype=np.float32)], axis=0)
        if 边界抹软像素 > 0:
            t = float(max(1, 边界抹软像素))
            e = np.exp((pool - pool.max(axis=0, keepdims=True)) / t)
            share = e / np.clip(e.sum(axis=0, keepdims=True), 1e-8, None)
            weight = share[:len(chosen)]
        else:
            weight = np.zeros_like(scores)
            win = pool.argmax(axis=0)
            ok = win < len(chosen)
            weight[win[ok], np.arange(n_img)[ok]] = 1.0

        # 整幅人像剪影：只让人身上的格子有归属，背景一格都不发滤镜
        mask_note = ""
        if mask_tensor is not None and not per_person:
            try:
                mm = mask_tensor.detach().cpu().float().numpy()
                if mm.ndim == 3:
                    mm = mm[0]
                ys = np.linspace(0, mm.shape[0], gh + 1).astype(int)
                xs = np.linspace(0, mm.shape[1], gw + 1).astype(int)
                pooled = np.array([[float(mm[a:b, c:d].mean()) for c, d in zip(xs[:-1], xs[1:])]
                                   for a, b in zip(ys[:-1], ys[1:])], dtype=np.float32)
                mf = pooled.ravel()
                weight = weight * mf[None, :]
                mask_note = ("人像遮罩已生效：身上 %d 格（%.0f%%），背景 %d 格不发任何滤镜" % (
                    int((mf > 0.5).sum()), 100.0 * (mf > 0.5).mean(), int((mf <= 0.5).sum())))
            except Exception as exc:
                mask_note = "⚠ 人像遮罩没能用上（%s），这一趟只按脸就近判" % type(exc).__name__

        owner_idx = np.where(scores.max(axis=0) <= 0, -1, weight.argmax(axis=0)).astype(np.int8)

        boxes = []
        for i, f in enumerate(chosen):
            nm = names[i] if i < len(names) else f"{i + 1}号"
            boxes.append({"owner": i + 1, "name": nm,
                          "x": int(f["cx"] - f["w"] / 2), "y": int(f["cy"] - f["h"] / 2),
                          "w": int(f["w"]), "h": int(f["h"])})

        region = {"grid_w": gw, "grid_h": gh, "n_img": n_img, "ds": DS,
                  "canvas": (int(画布宽), int(画布高)), "boxes": boxes,
                  "weight": torch.from_numpy(np.ascontiguousarray(weight)),
                  "owner": torch.from_numpy(np.ascontiguousarray(owner_idx)),
                  "feather": int(边界抹软像素), "mode": "人脸就近"}

        lines = list(env_lines) + [
                 f"草稿图尺寸 {draft.shape[2]}×{draft.shape[1]}，检测到 {len(dfaces)} 张脸，"
                 f"按角色名字算 {n_people} 个人"
                 + (f"，参考脸 {min(n_people, refs.shape[0]) if refs is not None else 0} 张" if refs is not None
                    else "，没接参考脸（只有体检分没有，地盘照分）"),
                 auto_note,
                 # 选了自动切却没切成（模型不认 / 切失败），这一行必须把原因顶上去，
                 # 不许退化成「不接遮罩」——那等于把话吞了
                 "遮罩来源：%s%s" % (mask_src if (auto_cut or mask_tensor is not None)
                                     else "不接遮罩（只按脸就近＋半径）",
                                     "" if mask_tensor is None else
                                     ("｜类型：%s｜已放大 %d 像素" %
                                      ("每人一块（硬边界）" if per_person else "整幅人像剪影", expand))),
                 sam_note,
                 (radius_note or "归属半径＝手填 %d 像素" % int(R)),
                 f"编号规则：{谁是谁}"
                 + ("（1 号＝草稿里最左边那张脸，从左到右升序）" if "左" in 谁是谁 else "（1 号＝最上面那张）")
                 + "；体检分只做参考，不参与分地盘", ""]
        if sam_guide:
            lines += [sam_guide, ""]
        for i, f in enumerate(chosen):
            nm = boxes[i]["name"]
            lines.append(f"人物{i + 1}（{nm}）中心 x={f['cx']:.0f} y={f['cy']:.0f} — {f['来源']}")
        if len(chosen) == 2:
            mid = (chosen[0]["cx"] + chosen[1]["cx"]) / 2
            lines.append("")
            lines.append(f"分界线自动落在 x≈{mid:.0f}（两个人中心正中间）")
            for i, f in enumerate(chosen):
                if abs(f["cx"] - mid) < f["w"] * 0.5:
                    lines.append(f"  ⚠ {boxes[i]['name']} 会跨过这条线，靠近的那半边拿不满自己的滤镜")
        elif len(chosen) > 2:
            lines.append("")
            lines.append("三个人以上：交界按「离谁的脸近归谁」自动分，不是等距切列")
        public = int((scores.max(axis=0) <= 0).sum())
        lines.append(f"不属于任何人的格子：{public} 格（归属半径 {R:.0f}）")
        if int(画布宽) % DS or int(画布高) % DS:
            lines.append("⚠ 画布宽/高不是 16 的倍数（%s×%s）：模型的小格子只能整除，"
                         "最后 %d 行 / %d 列拿不到任何滤镜。建议改成 %d×%d。" % (
                             画布宽, 画布高, int(画布高) - gh * DS, int(画布宽) - gw * DS,
                             int(画布宽) // DS * DS, int(画布高) // DS * DS))
        if body_note:
            lines.append(body_note)
        if mask_note:
            lines.append(mask_note)
        if ref_warn:
            lines += [""] + [f"体检提示：{w}" for w in ref_warn]
        if notes:
            lines.append("")
            lines.append("需要留意：")
            lines += [f"  {n}" for n in notes]
        if len(dfaces) > n_people:
            lines.append(f"注意：草稿里还有 {len(dfaces) - n_people} 张脸没被指派（可能多出了人）")

        Hi, Wi = int(draft.shape[1]), int(draft.shape[2])   # 草稿的真实像素尺寸
        W, H = Wi, Hi
        # 网格→图片按比值映射，不用整数倍 repeat：画布高不是 16 的倍数时，
        # 整数倍会少算最后几行，导致相加时形状对不上
        ri = np.minimum((np.arange(H) * gh) // H, gh - 1)
        ci = np.minimum((np.arange(W) * gw) // W, gw - 1)
        up = weight.reshape(weight.shape[0], gh, gw)[:, ri][:, :, ci]
        palette = [(0.20, 0.75, 0.45), (0.30, 0.55, 0.95), (0.95, 0.65, 0.20),
                   (0.85, 0.40, 0.55), (0.60, 0.40, 0.90), (0.35, 0.80, 0.80)]
        base = _to_bgr(草稿图)[0:1].astype(np.float32)[:, :, :, ::-1] / 255.0
        if base.shape[1] != H or base.shape[2] != W:
            ys = np.linspace(0, base.shape[1] - 1, H).astype(np.int32)
            xs = np.linspace(0, base.shape[2] - 1, W).astype(np.int32)
            base = base[:, ys][:, :, xs]
        overlay = np.zeros((H, W, 3), dtype=np.float32)
        for i in range(weight.shape[0]):
            rgb = np.array(palette[i % len(palette)], dtype=np.float32)
            m = np.clip(up[i], 0.0, 1.0)
            for c in range(3):
                overlay[..., c] += m * rgb[c]
        acc = np.clip(up.sum(axis=0), 0.0, 1.0)[..., None]
        overlay = overlay / np.clip(acc, 1e-6, None)
        a = np.clip(0.45 * acc, 0.0, 1.0)
        out = base * (1 - a) + overlay[None, ...] * a

        # 把编号画成大框 + 大号数字：滤镜节点里填几号，就对应这里几号那张脸
        s = max(4, int(min(H, W) * 0.007))
        dark = np.array([0.02, 0.02, 0.02], dtype=np.float32)
        cv = out[0]
        for i, b in enumerate(boxes):
            bright = np.clip(np.array(palette[i % len(palette)], dtype=np.float32) * 0.55 + 0.45, 0, 1)
            x0, y0, x1, y1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
            _rect(cv, x0 - 3 * s, y0 - 3 * s, x1 + 3 * s, y1 + 3 * s, dark, t=2 * s)
            _rect(cv, x0 - 2 * s, y0 - 2 * s, x1 + 2 * s, y1 + 2 * s, bright, t=s)
            ly = y0 - 2 * s - 9 * s
            if ly < 0:
                ly = y1 + 3 * s
            _put_text(cv, str(i + 1), x0 - 2 * s, ly, bright, s=s, bg=dark)
        vis = torch.from_numpy(np.ascontiguousarray(out)).float()

        return (region, "\n".join(lines), vis)


NODE_CLASS_MAPPINGS = {"MOSSIdFaceRegionMap": MOSSIdFaceRegionMap}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSIdFaceRegionMap": "人脸定位地盘"}
