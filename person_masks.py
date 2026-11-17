"""逐人身体遮罩（SAM2 点提示版）

现成的 SAM2 节点大多只输出**合并遮罩**——几个人并成一块，下游拿到后
会把没被分到的人判成"没有遮罩"，身份直接失效。
这里直接用官方 sam2 包的图片预测器，**从每张脸向四周撒点**，一人一块遮罩，
于是：
  · 每个人的地盘只能在自己那块身体里 —— 站着女孩的腰和脚不再被邻居瓜分
  · 观察者的手不属于任何一个人的连通对象 —— 拿不到滤镜

本文件还提供「这台机器上有哪些 SAM 模型」的清单（scan_sam_models），
「人脸定位地盘」和这里共用同一份，避免出现两个长得不一样的下拉。

只读模型文件夹，不修改 ComfyUI 本体和任何权重。
"""

import glob
import os

import numpy as np
import torch


def _sc():
    global _SC
    if _SC is None:
        try:
            from . import self_check as _m
        except Exception:
            import importlib.util as _iu
            _spec = _iu.spec_from_file_location(
                "self_check", os.path.join(os.path.dirname(os.path.abspath(__file__)), "self_check.py"))
            _m = _iu.module_from_spec(_spec)
            _spec.loader.exec_module(_m)
        _SC = _m
    return _SC


_SC = None

DS = 16

# ── 这台机器上到底有哪些 SAM 模型 ──────────────────────────────────────────
#
# 下拉里每一项后面都跟着「这一项本插件能不能直接用它自动切」，
# 三种结论：
#   内置可用    —— 官方 SAM2 的 .pt，本插件自己就能加载
#   需外接节点  —— SAM3 一类，本插件不调用它，但你可以用别的插件的
#                  SAM 节点出遮罩后接进「人脸定位地盘」的『遮罩』口
#   格式不支持  —— 别的插件转换过的 .safetensors / mobile_sam / sam_hq 等，
#                  官方加载器不认，这里只列出来让你知道本机有什么
TAG_OK = "内置可用"
TAG_FETCH = "内置·没有就自动下载"
TAG_EXT = "需外接节点"
TAG_NO = "格式不支持"
SEP = " ｜ "


def _models_root():
    try:
        return _sc().MODELS
    except Exception:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "models")


def _sam_dirs():
    """要扫哪些文件夹：ComfyUI 登记过的 sam2/sam3/sam，再加 models 下的同名目录。
    Windows 大小写不敏感，所以 normcase 去重。"""
    roots = []
    try:
        import folder_paths
        for key in ("sam2", "sam3", "sam"):
            try:
                roots += list(folder_paths.get_folder_paths(key) or [])
            except Exception:
                pass
    except Exception:
        pass
    m = _models_root()
    for name in ("SAM2", "SAM3", "SAM"):
        roots.append(os.path.join(m, name))
    out, seen = [], set()
    for r in roots:
        if not r:
            continue
        k = os.path.normcase(os.path.abspath(r))
        if k in seen or not os.path.isdir(r):
            continue
        seen.add(k)
        out.append(r)
    return out, m


def _tag_of(fname, path):
    """按文件名和位置判断这个权重本插件能不能自己用"""
    low = fname.lower()
    if "sam3" in low or os.path.normcase(os.path.dirname(path)).endswith("sam3"):
        return TAG_EXT, "本插件不调用 SAM3；用带 SAM3 的节点出遮罩后接『遮罩』口"
    if "sam2" in low and low.endswith(".pt"):
        return TAG_OK, "官方 SAM2 的 .pt，本插件直接加载"
    return TAG_NO, "不是官方 SAM2 的 .pt（多半是别的插件转过的格式），官方加载器不认"


def scan_sam_models():
    """返回该机器上所有能看到的 SAM 权重，按「内置可用 → 需外接 → 格式不支持」排。

    每项是字典：标签（下拉里显示的那一行）/ 相对路径 / 绝对路径 / 大小MB / 结论 / 说明
    本插件自带的默认权重即使还没下载，也永远排在第一位（标成『没有就自动下载』）。
    """
    m = _models_root()
    default_rel = os.path.join("SAM2", "sam2.1_hiera_base_plus.pt")
    rows, seen = [], set()
    for d, _root in [(d, None) for d in _sam_dirs()[0]]:
        for ext in ("*.pt", "*.safetensors", "*.onnx", "*.pth"):
            for p in sorted(glob.glob(os.path.join(d, ext))):
                fn = os.path.basename(p)
                rel = os.path.relpath(p, m) if _inside(p, m) else os.path.join("其它目录", fn)
                key = os.path.normcase(rel)
                if key in seen:
                    continue
                seen.add(key)
                tag, why = _tag_of(fn, p)
                try:
                    mb = os.path.getsize(p) / 1048576.0
                except Exception:
                    mb = 0.0
                rows.append({"相对路径": rel.replace("\\", "/"), "绝对路径": p,
                             "大小MB": mb, "结论": tag, "说明": why,
                             "标签": tag + SEP + rel.replace("\\", "/") + SEP + "%.0f MB" % mb})
    want = {TAG_OK: 0, TAG_EXT: 1, TAG_NO: 2}
    rows.sort(key=lambda r: (want.get(r["结论"], 3), r["相对路径"]))
    if not any(os.path.normcase(r["相对路径"]) == os.path.normcase(default_rel.replace("\\", "/"))
               for r in rows):
        rows.insert(0, {"相对路径": default_rel.replace("\\", "/"),
                        "绝对路径": os.path.join(m, default_rel),
                        "大小MB": SAM2_MB, "结论": TAG_OK,
                        "说明": "本插件默认用的权重，第一次用到时自动下载",
                        "标签": TAG_FETCH + SEP + default_rel.replace("\\", "/") + SEP
                                + "%.0f MB" % SAM2_MB})
    return rows


SAM2_MB = 308.6


def _inside(p, root):
    try:
        return os.path.normcase(os.path.abspath(p)).startswith(os.path.normcase(os.path.abspath(root)))
    except Exception:
        return False


def sam_labels():
    """下拉用的一串标签。永远不为空（至少有默认那一项）。"""
    try:
        ls = [r["标签"] for r in scan_sam_models()]
    except Exception:
        ls = []
    return ls or [TAG_FETCH + SEP + "SAM2/sam2.1_hiera_base_plus.pt" + SEP + "%.0f MB" % SAM2_MB]


def sam_default_label():
    ls = sam_labels()
    for t in (TAG_OK, TAG_FETCH):
        for l in ls:
            if l.startswith(t):
                return l
    return ls[0]


def sam_rel(label):
    """从下拉标签里取出相对路径；标签被手改过也尽量兜住"""
    s = str(label or "")
    parts = [x.strip() for x in s.split(SEP)]
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return parts[0]


def sam_row(label):
    rel = sam_rel(label)
    rows = scan_sam_models()
    want = os.path.normcase(rel)
    for r in rows:
        if os.path.normcase(r["相对路径"]) == want:
            return r
    # 下拉标签可能只存了文件名（没有目录前缀），按文件名再兜一次
    base = os.path.normcase(os.path.basename(rel))
    for r in rows:
        if os.path.normcase(os.path.basename(r["相对路径"])) == base:
            return r
    return {"相对路径": rel, "绝对路径": os.path.join(_models_root(), rel),
            "大小MB": 0.0, "结论": TAG_NO, "说明": "清单里没找到这一项", "标签": str(label)}


def sam_path(label):
    """把下拉标签换成绝对路径；默认权重不在磁盘上时先自动下载"""
    row = sam_row(label)
    p = row["绝对路径"]
    if os.path.exists(p):
        return p
    if row["结论"] == TAG_OK and "sam2" in row["相对路径"].lower():
        ok, bad = _sc().fetch_weights([("SAM2 检查点", p, _sc().SAM2_URL, _sc().SAM2_SIZE, "")])
        if os.path.exists(p):
            return p
        raise RuntimeError((bad[0] if bad else "下载结束但文件不在") +
                           "\n退路：手动下载 %s 放到 %s" % (_sc().SAM2_URL, os.path.dirname(p)))
    raise RuntimeError("%s 在磁盘上找不到（%s）" % (row["相对路径"], p))


# ── 官方 sam2 包的加载器 ────────────────────────────────────────────────────

_CACHE = {}


def _predictor(ckpt, device):
    """ckpt 必须是绝对路径（用 sam_path() 从下拉标签换出来）。同一路径只加载一次。"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    key = (ckpt, device)
    if key not in _CACHE:
        base = os.path.splitext(os.path.basename(ckpt))[0]
        # 官方包里的配置文件名用的是 b+ 而不是 base_plus，得先换算一次
        cfgname = base.replace("base_plus", "b+").replace("_fp16", "").replace("-fp16", "")
        ver = "sam2.1" if "sam2.1" in base else "sam2"
        cands = ["configs/%s/%s.yaml" % (ver, cfgname),
                 "configs/%s/%s.yaml" % ("sam2" if ver == "sam2.1" else "sam2.1",
                                         cfgname.replace("sam2.1", "sam2") if ver == "sam2.1"
                                         else cfgname.replace("sam2", "sam2.1", 1)),
                 "%s.yaml" % cfgname]
        last = None
        for cfg in cands:
            try:
                model = build_sam2(cfg, ckpt, device=device)
                _CACHE[key] = SAM2ImagePredictor(model)
                break
            except Exception as exc:
                last = "%s ← %s" % (cfg, str(exc).splitlines()[0][:120])
        else:
            raise RuntimeError("配置文件试遍都加载不了，最后一次：%s" % last)
    return _CACHE[key]


def _pt_in(mask, x, y):
    h, w = mask.shape
    xi = int(min(w - 1, max(0, round(float(x)))))
    yi = int(min(h - 1, max(0, round(float(y)))))
    return bool(mask[yi, xi])


def _face_cover(mask, f):
    """这块遮罩盖住了本人脸框的几成（0~1）"""
    H, W = mask.shape
    fw, fh = max(1.0, float(f["w"])), max(1.0, float(f["h"]))
    x0 = int(max(0, f["cx"] - fw * 0.36)); x1 = int(min(W, f["cx"] + fw * 0.36))
    y0 = int(max(0, f["cy"] - fh * 0.36)); y1 = int(min(H, f["cy"] + fh * 0.36))
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(mask[y0:y1, x0:x1].mean())


def _pick_candidate(cands, own, others, H, W):
    """从 SAM 的三档候选里挑"这个人的完整身体"：

    判据只有两条——**必须含本人脸心**、**不许含别人的脸心**。
    SAM 本身不认识"谁是谁"，这两条就是我们把"身份"补上的方式。
    在这两条之内取面积最大的那块：单点提示时 SAM 常给"脸/头/半身/全身"三档，
    过去按"脸正下方"猜身体位置，人一躺下就全错了；按面积取最大，
    躺、趴、弯腰都不再依赖姿态假设。
    """
    best = None
    cap = 0.75 * H * W
    for mm in cands:
        mk = np.asarray(mm) > 0
        if mk.shape[:2] != (H, W):
            continue
        if not _pt_in(mk, own["cx"], own["cy"]):
            continue
        stolen = False
        for g in others:
            if _pt_in(mk, g["cx"], g["cy"]):
                stolen = True
                break
        if stolen:
            continue
        if mk.sum() > cap:
            continue
        if best is None or mk.sum() > best.sum():
            best = mk
    return best


def _probe_points(mask, own):
    """从当前遮罩的最外缘，沿 16 个方向各往外探一格（自己身体探到哪就接着往哪走）"""
    H, W = mask.shape
    if not mask.any():
        return [(own["cx"], own["cy"])]
    scale = max(float(own["w"]), float(own["h"]), 8.0)
    ys, xs = np.nonzero(mask)
    ang = np.arctan2(ys - own["cy"], xs - own["cx"])
    dist = np.hypot(xs - own["cx"], ys - own["cy"])
    pts = []
    for k in range(16):
        a = 2.0 * np.pi * k / 16.0
        sel = np.abs(((ang - a + np.pi) % (2.0 * np.pi)) - np.pi) < (np.pi / 16.0)
        if not sel.any():
            continue
        d = float(dist[sel].max()) + 0.55 * scale
        px = own["cx"] + d * np.cos(a)
        py = own["cy"] + d * np.sin(a)
        if 2 <= px < W - 2 and 2 <= py < H - 2:
            pts.append((px, py))
    return pts


def _star_mask(pred, H, W, own, others, rounds=4):
    """星芒探测 + 逐轮外扩：向外撒点 → 过身份校验 → 收下 → 从新外缘继续

    每轮把 16 个探点一次性交给 SAM（批量提示），一轮约 0.2 秒。
    连续一轮什么都收不下就停——多数图第一轮就已经把整个人拿全了，
    外扩只是兜住"第一刀只切到半身"的情况。
    """
    cur = np.zeros((H, W), bool)
    got = 0
    with torch.no_grad():
        for _ in range(max(1, rounds)):
            pts = _probe_points(cur, own)
            if not pts:
                break
            k = len(pts)
            pc = np.array(pts, np.float32)[:, None, :]
            pl = np.ones((k, 1), np.int32)
            ms, _scores, _ = pred.predict(point_coords=pc, point_labels=pl, multimask_output=True)
            ms = np.asarray(ms)
            cand_batches = [ms] if (k == 1 and ms.ndim == 3) else [ms[b] for b in range(k)]
            acc = 0
            for b in range(k):
                mk = _pick_candidate(cand_batches[b], own, others, H, W)
                if mk is not None:
                    cur = cur | mk
                    acc += 1
            got += acc
            if acc == 0:
                break
    return cur, got


def _legacy_face_down_mask(pred, img_rgb, faces, i, H, W):
    """旧的"脸下方/左下/右下"点提示 + 框提示。只在星芒一刀都没切到自己脸时兜底。"""
    f = faces[i]
    fx, fy, fw, fh = f["cx"], f["cy"], f["w"], f["h"]
    sets = [
        [(0.0, 1.3), (0.0, 2.2)],
        [(-0.9, 1.6), (-1.6, 2.4)],
        [(0.9, 1.6), (1.6, 2.4)],
        [(0.0, 1.6), (-1.5, 2.6), (1.5, 2.6)],
    ]
    pool = []
    for cand in sets:
        pts, lab = [[fx, fy]], [1]
        for dx, dy in cand:
            _x, _y = fx + dx * fw, fy + dy * fh
            if 0 <= _x < W and 0 <= _y < H:
                pts.append([_x, _y]); lab.append(1)
        if len(pts) < 2:
            continue
        for j, g in enumerate(faces):
            if j != i:
                pts.append([g["cx"], g["cy"]]); lab.append(0)
        m, _s, _ = pred.predict(point_coords=np.array(pts, np.float32),
                               point_labels=np.array(lab), multimask_output=True)
        for mm in m:
            mk = mm > 0
            if not _pt_in(mk, fx, fy):
                continue
            if any(_pt_in(mk, g["cx"], g["cy"]) for j, g in enumerate(faces) if j != i):
                continue
            pool.append(mk)
    if not pool:
        return None
    return np.logical_or.reduce(pool)


def predict_person_masks(img_rgb, faces, ckpt, device="cuda"):
    """给「人脸定位地盘」内部调用：输入 RGB uint8 图和已排好序的脸列表，输出每人一块 0/1 遮罩列表。

    faces 里每项要有 cx, cy, w, h。ckpt 传下拉里那一整行标签（会自动换算成路径，
    默认权重不在磁盘上时先自动下载）。失败时抛异常，由调用方决定怎么退回。

    做法（2026-11 改）：**星芒探测 + 脸框归属校验 + 逐轮外扩**。
    不再假设"身体在脸正下方"——躺着、趴着、打横时那个假设会让人整块切错，
    甚至把下面那个人的脸吃进来。现在只做一件事：
    从脸心往外 16 个方向撒点，只收"含本人脸心、不含别人脸心"的最大候选，
    收下后从新外缘继续撒，直到不再长大。
    """
    import torch as _torch

    pred = _predictor(sam_path(ckpt), device)
    H, W = img_rgb.shape[:2]
    out = []
    with _torch.no_grad():
        pred.set_image(img_rgb)
        for i, f in enumerate(faces):
            own = {"cx": float(f["cx"]), "cy": float(f["cy"]),
                   "w": float(f["w"]), "h": float(f["h"])}
            others = [{"cx": float(g["cx"]), "cy": float(g["cy"])}
                      for j, g in enumerate(faces) if j != i]
            mk, got = _star_mask(pred, H, W, own, others)
            if got == 0 or _face_cover(mk, own) < 0.30:
                # 兜底：星芒一个候选都没通过校验（极少见），回到旧的"脸下方"点提示
                leg = _legacy_face_down_mask(pred, img_rgb, faces, i, H, W)
                if leg is not None:
                    mk = np.logical_or(mk, leg) if mk.any() else leg
            # 只保留"和本人脸连通"的那一块：SAM 偶尔会把别人的鞋、手粘进来
            try:
                import cv2 as _cv
                _n, lab, _st, _c = _cv.connectedComponentsWithStats(mk.astype(np.uint8), connectivity=8)
                keep = int(lab[int(min(H - 1, max(0, own["cy"]))), int(min(W - 1, max(0, own["cx"])))])
                if keep > 0:
                    mk = (lab == keep)
            except Exception:
                pass
            out.append(mk.astype(np.float32))
    return out


def sam_is_builtin(label):
    """这一项本插件能不能自己拿去切遮罩"""
    row = sam_row(label)
    return row["结论"] in (TAG_OK, TAG_FETCH), row


class MOSSIdPersonMasks:
    """逐人身体遮罩：每张脸往四周撒点，SAM 各切一块"""

    @classmethod
    def INPUT_TYPES(cls):
        ls = sam_labels()
        return {
            "required": {
                "草稿图": ("IMAGE", {"tooltip": "第一趟出的草稿（不带人物滤镜那张）"}),
                "人数上限": ("INT", {"default": 12, "min": 1, "max": 12, "step": 1, "tooltip": '最多处理几个人，防止画面里出现很多无关小脸时切出一堆块。'}),
                "谁是谁": (["按位置排（左到右）", "按位置排（上到下）"],
                          {"default": "按位置排（左到右）",
                           "tooltip": "遮罩的顺序＝下游「人脸定位地盘」的 1 号、2 号、3 号…默认最左边的人＝1 号"}),
                "运行设备": (["cuda", "cpu"], {"default": "cuda", "tooltip": '有显卡选 cuda（快），显存不够或没有显卡选 cpu（慢一些，结果一致）。'}),
                "SAM模型": (ls, {"default": sam_default_label(),
                                "tooltip": "这台机器上能看到的 SAM 权重都在这里，一行三段："
                                           "能不能直接用 ｜ 放在哪 ｜ 多大。\n"
                                           "『内置可用』是本插件自己就能加载的官方 SAM2 .pt；"
                                           "默认的 sam2.1_hiera_base_plus 如果还没下载，第一次用会自动下（约 309MB，只一次）。\n"
                                           "标了『需外接节点』『格式不支持』的这一档切不出来，"
                                           "用带该模型的节点出遮罩后接进「人脸定位地盘」的『遮罩』口即可。"}),
            },
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("每人一块遮罩", "说明")
    FUNCTION = "build"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "以每张脸为起点向四周撒点，只收「含本人脸、不含别人脸」的那块，逐人切出身体遮罩"

    def build(self, 草稿图, 人数上限, 谁是谁, 运行设备, SAM模型):
        ok, row = sam_is_builtin(SAM模型)
        if not ok:
            raise RuntimeError(
                "「%s」本插件加载不了（%s）。\n"
                "想用这一档的话：用带这个模型的节点（例如 ComfyUI-RMBG 的 SAM3Segment）"
                "以人为提示生成遮罩，把结果接到「人脸定位地盘」的『遮罩』口，"
                "并把那边的『遮罩来源』改成『外部接入』。\n"
                "清单：%s" % (row["相对路径"], row["说明"],
                              "、".join(r["相对路径"] + "（" + r["结论"] + "）"
                                       for r in scan_sam_models() if r["结论"] in (TAG_OK, TAG_FETCH))))
        face_region = None
        try:
            from . import face_region as _fr            # 插件按包加载时走相对导入
        except Exception:
            _fr = None
        if _fr is None:
            try:
                import face_region as _fr               # 单独调试时走绝对导入
            except Exception:
                import importlib.util as _iu
                _spec = _iu.spec_from_file_location(
                    "face_region", os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_region.py"))
                _fr = _iu.module_from_spec(_spec)
                _spec.loader.exec_module(_fr)
        _faces, _to_bgr = _fr._faces, _fr._to_bgr       # 复用同一套检脸，保证编号一致

        img = 草稿图[0].float().cpu().numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        faces = _faces(_to_bgr(草稿图))
        if not faces:
            raise RuntimeError("草稿里一张脸都没检到，切不出人物遮罩。")
        keyf = (lambda d: d["cx"]) if "左" in 谁是谁 else (lambda d: d["cy"])
        faces = sorted(faces, key=keyf)[:max(1, int(人数上限))]

        try:
            ckpt = sam_path(SAM模型)
            pred = _predictor(ckpt, 运行设备)
        except Exception as exc:
            raise RuntimeError(
                "SAM 权重没加载起来：%s\n要的是官方 SAM2 的 .pt（放在 models/SAM2），"
                "文件夹里的 .safetensors 多半是别的插件转过的格式，官方加载器不认。" % exc)

        lines = ["逐人切遮罩：模型 %s（%s）｜设备 %s｜检到 %d 张脸 → 出 %d 块（从脸心向四周撒点，"
                 "只收含本人脸、不含别人脸的最大一块，再从外缘继续撒）" % (
                     row["相对路径"], row["结论"], 运行设备, len(faces), len(faces)), ""]
        # 复用和人脸定位地盘同一套实现（星芒撒点 + 脸框校验 + 连通域约束），
        # 避免"改了一边另一边没变"。
        masks = predict_person_masks(img, faces, SAM模型, 运行设备)
        for i, (f, mk) in enumerate(zip(faces, masks)):
            ys, xs = np.nonzero(mk > 0.5)
            cx = xs.mean() if len(xs) else 0
            cy = ys.mean() if len(ys) else 0
            lines.append("第%d块（%s号）：面积 %.1f%%｜质心 x=%.0f y=%.0f｜离该脸中心 %.0f 像素"
                         "｜自己脸心在块内=%s" % (
                             i + 1, i + 1, 100.0 * float((mk > 0.5).mean()), cx, cy,
                             float(np.hypot(cx - f["cx"], cy - f["cy"])),
                             "是" if mk[int(f["cy"]), int(f["cx"])] > 0.5 else "否"))

        out = torch.from_numpy(np.stack(masks)).float()

        # 提醒重叠：两块遮罩大面积重合，说明 SAM2 把两个人当成一个对象了
        big = []
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                inter = float((masks[i] * masks[j]).sum())
                union = float(((masks[i] + masks[j]) > 0).sum())
                if union > 0 and inter / union > 0.35:
                    big.append("%d号与%d号重合 %.0f%%" % (i + 1, j + 1, 100.0 * inter / union))
        if big:
            lines += ["", "⚠ 有人的遮罩高度重合（SAM2 把他们当一个对象了）：", "   " + "；".join(big)]
        lines += ["", "接法：这块遮罩 → 「人脸定位地盘」的『遮罩』输入（多张＝一人一块）；"
                      "那边的『遮罩来源』选『外部接入』，『归属半径』填 0 即可（有遮罩时半径只兜底）。"
                      "不想多接一个节点的话，直接把『遮罩来源』选成自动逐人切，效果一样。"]
        return (out, "\n".join(lines))


NODE_CLASS_MAPPINGS = {"MOSSIdPersonMasks": MOSSIdPersonMasks}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSIdPersonMasks": "逐人身体遮罩·SAM2"}


def split_by_silhouette(img_rgb, faces, silhouette):
    """用"整幅人物剪影 + 每张脸的位置"分出每人一块

    做法：以每张脸为种子，在**剪影内部**做漫水（watershed，跟着图像边缘走），
    然后把结果限制在剪影里。这样弯腰、坐、蹲都不影响——只要人整体被剪影圈住，
    身体就会连到自己的脸。
    返回：每人的 0/1 遮罩列表；分不出来时返回 None（调用方退回原逻辑）。
    """
    import numpy as _np
    try:
        import cv2 as _cv
    except Exception:
        return None
    sil = _np.asarray(silhouette, _np.float32)
    if sil.ndim == 3:
        sil = sil[0]
    H, W = img_rgb.shape[:2]
    if sil.shape[:2] != (H, W):
        sil = _cv.resize(sil, (W, H), interpolation=_cv.INTER_LINEAR)
    sil = (sil > 0.5).astype(_np.uint8)

    markers = _np.zeros((H, W), _np.int32)
    seeds = 0
    for i, f in enumerate(faces):
        cx, cy = int(f["cx"]), int(f["cy"])
        if not (0 <= cx < W and 0 <= cy < H):
            continue
        if sil[cy, cx] == 0:                     # 脸不在剪影里（剪影没圈到头）→ 跳过
            continue
        r = max(3, int(f["w"] * 0.18))
        _cv.circle(markers, (cx, cy), r, i + 1, -1)
        seeds += 1
    if seeds == 0:
        return None

    bgr = _cv.cvtColor(img_rgb, _cv.COLOR_RGB2BGR)
    mk = markers.copy()
    try:
        _cv.watershed(bgr, mk)
    except Exception:
        return None

    out = []
    for i in range(len(faces)):
        m = ((mk == i + 1) & (sil > 0)).astype(_np.float32)
        out.append(m)
    # 至少要有两块非空，否则说明没分开
    if sum(1 for m in out if m.sum() > 0) < 2 and len(faces) > 1:
        return None
    return out


def combine_per_person(img_rgb, masks, faces):
    """把任意一组遮罩整理成"每人一块"

    规则：
      1) 一块里只有一张脸 → 整块归他
      2) 一块里有多张脸 → 在这块内部按脸切开（watershed，沿图像边缘走）
      3) 一块里没有脸 → 并给"质心最近"的那个人（限 3 倍脸宽内），否则丢弃
    返回：与 faces 等长的 0/1 遮罩列表
    """
    import numpy as _np
    try:
        import cv2 as _cv
    except Exception:
        return None
    H, W = img_rgb.shape[:2]
    n = len(faces)
    if n == 0 or not masks:
        return None
    bm = [(_np.asarray(m, _np.float32) > 0.5) for m in masks]
    own = [_np.zeros((H, W), _np.uint8) for _ in range(n)]
    shared = []
    orphans = []
    for m in bm:
        inside = [i for i, f in enumerate(faces)
                  if 0 <= int(f["cy"]) < m.shape[0] and 0 <= int(f["cx"]) < m.shape[1]
                  and m[int(f["cy"]), int(f["cx"])]]
        if len(inside) == 0:
            orphans.append(m)
        elif len(inside) == 1:
            own[inside[0]] = _np.maximum(own[inside[0]], m.astype(_np.uint8))
        else:
            shared.append((m, inside))

    # 多脸块：并起来一起做分水岭，只在这块内部切
    if shared:
        union = _np.zeros((H, W), _np.uint8)
        all_inside = set()
        for m, inside in shared:
            union = _np.maximum(union, m.astype(_np.uint8))
            all_inside.update(inside)
        markers = _np.zeros((H, W), _np.int32)
        for i in sorted(all_inside):
            f = faces[i]
            _cv.circle(markers, (int(f["cx"]), int(f["cy"])),
                       max(3, int(f["w"] * 0.18)), i + 1, -1)
        bgr = _cv.cvtColor(img_rgb, _cv.COLOR_RGB2BGR)
        mk = markers.copy()
        try:
            _cv.watershed(bgr, mk)
            for i in sorted(all_inside):
                part = ((mk == i + 1) & (union > 0)).astype(_np.uint8)
                own[i] = _np.maximum(own[i], part)
        except Exception:
            # 分水岭失败：退回"离谁脸近"
            yy, xx = _np.mgrid[0:H, 0:W]
            d = _np.stack([_np.hypot(xx - faces[i]["cx"], yy - faces[i]["cy"]) for i in sorted(all_inside)])
            who = d.argmin(axis=0)
            for k, i in enumerate(sorted(all_inside)):
                own[i] = _np.maximum(own[i], (union * (who == k)).astype(_np.uint8))

    # 没有脸的块：并给质心最近的人
    for m in orphans:
        ys, xs = _np.where(m)
        if not len(xs):
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        cand = [(_np.hypot(f["cx"] - cx, f["cy"] - cy), i) for i, f in enumerate(faces)]
        dist, i = min(cand)
        if dist <= 3.0 * max(24.0, faces[i]["w"]):
            own[i] = _np.maximum(own[i], m.astype(_np.uint8))

    if not any(o.any() for o in own):
        return None
    return [o.astype(_np.float32) for o in own]
