"""逐人身体遮罩（SAM2 点提示版）

现成的 SAM2 节点大多只输出**合并遮罩**——几个人并成一块，下游拿到后
会把没被分到的人判成"没有遮罩"，身份直接失效。
这里直接用官方 sam2 包的图片预测器，**每张脸一个点提示**，一人一块遮罩，
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


def predict_person_masks(img_rgb, faces, ckpt, device="cuda"):
    """给「人脸定位地盘」内部调用：输入 RGB uint8 图和已排好序的脸列表，输出每人一块 0/1 遮罩列表。

    faces 里每项要有 cx, cy, w, h。ckpt 传下拉里那一整行标签（会自动换算成路径，
    默认权重不在磁盘上时先自动下载）。失败时抛异常，由调用方决定怎么退回。
    """
    import numpy as _np
    import torch as _torch

    pred = _predictor(sam_path(ckpt), device)
    out = []
    with _torch.no_grad():
        pred.set_image(img_rgb)
        for i, f in enumerate(faces):
            pts = [[f["cx"], f["cy"]], [f["cx"], f["cy"] + f["h"] * 1.6]]
            lab = [1, 1]
            for j, g in enumerate(faces):
                if j != i:
                    pts.append([g["cx"], g["cy"]])
                    lab.append(0)
            m, _scores, _ = pred.predict(point_coords=_np.array(pts, dtype=_np.float32),
                                         point_labels=_np.array(lab),
                                         multimask_output=True)
            areas = _np.array([float((mm > 0).mean()) for mm in m])
            k = int(_np.argmax(areas))
            mk = (m[k] > 0).astype(_np.float32)
            cy = int(min(mk.shape[0] - 1, max(0, f["cy"])))
            cx = int(min(mk.shape[1] - 1, max(0, f["cx"])))
            if mk[cy, cx] < 0.5:                      # 最大块不含自己的脸 → 退回含脸的那档
                for kk in _np.argsort(-areas):
                    cand = (m[kk] > 0).astype(_np.float32)
                    if cand[cy, cx] > 0.5:
                        mk = cand
                        break
            out.append(mk)
    return out


def sam_is_builtin(label):
    """这一项本插件能不能自己拿去切遮罩"""
    row = sam_row(label)
    return row["结论"] in (TAG_OK, TAG_FETCH), row


class MOSSIdPersonMasks:
    """逐人身体遮罩：每张脸一个点，SAM 各切一块"""

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
                "加下巴点": (["是", "否"], {"default": "是",
                                          "tooltip": "除了脸中心，再在脸的下方加一个提示点，"
                                                     "SAM2 更容易把整个人（含躯干）圈进来"}),
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
    DESCRIPTION = "以每张脸为提示点，用 SAM2 逐人切出身体遮罩，供「人脸定位地盘」的人物遮罩输入"

    def build(self, 草稿图, 人数上限, 谁是谁, 加下巴点, 运行设备, SAM模型):
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

        lines = ["逐人切遮罩：模型 %s（%s）｜设备 %s｜检到 %d 张脸 → 出 %d 块（取最大块＋别人的脸当反面）" % (
            row["相对路径"], row["结论"], 运行设备, len(faces), len(faces)), ""]
        masks = []
        chin = 加下巴点 == "是"
        with torch.no_grad():
            pred.set_image(img)
            for i, f in enumerate(faces):
                # 正面点：脸中心（可选再加胸口）；反面点：别人的脸（防止把邻居圈进来）
                pts = [[f["cx"], f["cy"]]]
                lab = [1]
                if chin:
                    pts.append([f["cx"], f["cy"] + f["h"] * 1.6])
                    lab.append(1)
                for j, g in enumerate(faces):
                    if j != i:
                        pts.append([g["cx"], g["cy"]])
                        lab.append(0)
                m, scores, _ = pred.predict(point_coords=np.array(pts, dtype=np.float32),
                                            point_labels=np.array(lab),
                                            multimask_output=True)
                # SAM2 一次给 3 个候选（小/中/大），要"整个人"就取面积最大的那块
                areas = np.array([float((mm > 0).mean()) for mm in m])
                k = int(np.argmax(areas))
                mk = (m[k] > 0).astype(np.float32)
                on_face = float(mk[max(0, min(mk.shape[0] - 1, int(f["cy"]))),
                                  max(0, min(mk.shape[1] - 1, int(f["cx"])))] > 0.5)
                if not on_face and areas.size > 1:      # 最大块不含自己的脸就退回中间那块
                    order = np.argsort(-areas)
                    for kk in order:
                        cand = (m[kk] > 0).astype(np.float32)
                        if cand[int(f["cy"]), int(f["cx"])] > 0.5:
                            mk, k = cand, int(kk)
                            break
                masks.append(mk)
                ys, xs = np.nonzero(mk > 0.5)
                cx = xs.mean() if len(xs) else 0
                cy = ys.mean() if len(ys) else 0
                lines.append("第%d块（%s号）：面积 %.1f%%｜质心 x=%.0f y=%.0f｜离该脸中心 %.0f 像素"
                             "｜候选第%d块（共3档，取最大）｜自己脸心在块内=%s" % (
                                 i + 1, i + 1, 100.0 * mk.mean(), cx, cy,
                                 float(np.hypot(cx - f["cx"], cy - f["cy"])), k + 1,
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
