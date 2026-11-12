"""环境自查（没有节点，插件内部用）

这个模块只在「人脸定位地盘」运行时被静默调用一次，做三件事：
  1 查必需项还在不在，不在就直接停下来告诉你省事的补法
  2 缺权重文件就自己从官方地址下回来（只在第一次发生）
  3 什么都齐 → 一个字都不说

为什么不做成节点：装依赖是插件自己的事，不该让使用者先去找一个
"体检节点"跑一遍才知道少东西。所以这里只保留函数，不注册节点。

分工（上架后新用户第一次装，谁负责什么）：
  · ComfyUI-Manager 读 requirements.txt → 装 Python 包（insightface 等）
  · ComfyUI-Manager 接着自动跑 install.py → 补 SAM2 权重
  · insightface 库第一次被调用时自己去官方发布页下人脸模型（antelopev2）
  · 以上任何一步没做成，本模块在节点运行时兜底再补一次权重
"""
import importlib.util
import os


def _models_dir():
    """优先用 ComfyUI 自己登记的模型根目录；脱离 ComfyUI 单独导入时按文件位置逐级
    往上找同时有 custom_nodes 和 models 的那一层。推算错了只会多提示一句，不会卡死流程"""
    try:
        import folder_paths
        d = getattr(folder_paths, "models_dir", None)
        if d and os.path.isdir(d):
            return d
    except Exception:
        pass
    here = os.path.abspath(__file__)
    for _ in range(6):
        here = os.path.dirname(here)
        if os.path.isdir(os.path.join(here, "custom_nodes")) and os.path.isdir(os.path.join(here, "models")):
            return os.path.join(here, "models")
    return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(here))), "models")


MODELS = _models_dir()
# insightface 的模型根。库会把 antelopev2 解压到 <这里>/models/antelopev2/
INSIGHTFACE_ROOT = os.path.join(MODELS, "insightface")

# (查什么包, 开的是哪个功能, 手动补法（退路）, 是不是必需)
PACKAGES = [
    ("insightface", "检脸、按位置编号、地盘可视化", "pip install insightface", True),
    ("onnxruntime", "上面那个人脸模型的运行库", "pip install onnxruntime", True),
    ("cv2", "遮罩胀开、把遮罩降到小格子、画序号框", "pip install opencv-python", True),
    ("torch", "所有节点的地基", "随 ComfyUI 安装", True),
    ("numpy", "所有节点的地基", "随 ComfyUI 安装", True),
    ("sam2", "自动逐人切遮罩这一档",
     "pip install git+https://github.com/facebookresearch/sam2.git", False),
]

# 权重：(名字, 相对 models 的路径, 官方地址, 期望字节数, 影响什么)
# 只列本插件需要自己下的。人脸模型不在这里——insightface 库自己会下，重复下反而多占空间。
WEIGHTS = [
    ("SAM2 检查点", os.path.join("SAM2", "sam2.1_hiera_base_plus.pt"),
     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
     323606802, "没有就切不出「每人一块」，其余功能照常"),
]

# 本插件自带的默认权重（缺失时会自动下载，所以清单里永远有它）
DEFAULT_SAM2 = WEIGHTS[0][1]
SAM2_URL = WEIGHTS[0][2]
SAM2_SIZE = WEIGHTS[0][3]


def one_click_fix():
    """缺东西时，真正省事的那一步只有一句。pip 命令只是退路。"""
    return ("省事的一步：ComfyUI-Manager → 本插件 → Reinstall（自动装包并补权重），"
            "或双击插件文件夹里的 install.bat，然后重启 ComfyUI。")


def missing_items():
    """返回 (缺的必需包, 缺的可选包, 缺的权重)。权重项是 (名字, 绝对路径, 地址, 字节数, 影响)"""
    need, opt, weights = [], [], []
    for mod, _purpose, _how, must in PACKAGES:
        if importlib.util.find_spec(mod) is not None:
            continue
        (need if must else opt).append(mod)
    for name, rel, url, size, impact in WEIGHTS:
        if not os.path.exists(os.path.join(MODELS, rel)):
            weights.append((name, os.path.join(MODELS, rel), url, size, impact))
    return need, opt, weights


def _rel(p):
    """能折算成相对 models 的短路径就用短的；算不出（例如跨盘）就退回绝对路径。
    这只是句提示语，不能因为它算不出来就把已经下好的文件报成下载失败。"""
    try:
        return "models/" + os.path.relpath(p, MODELS).replace("\\", "/")
    except Exception:
        return p


def fetch_weights(weights):
    """下载缺失的权重。返回 (成功说明列表, 失败说明列表)。只碰权重，不碰 Python 包。"""
    import shutil
    import sys
    import urllib.request

    ok, bad = [], []
    for name, dst, url, size, _impact in weights:
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".part"
            req = urllib.request.Request(url, headers={"User-Agent": "comfyui-moss-multiid"})
            total = 0
            with urllib.request.urlopen(req, timeout=1800) as r, open(tmp, "wb") as fh:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    total += len(chunk)
                    sys.stdout.write("\r[MOSS多身份] 正在下载 %s：%.0f%%\n" % (
                        name, 100.0 * total / size if size else 0.0))
                    sys.stdout.flush()
            if size and total < size * 0.98:
                os.remove(tmp)
                raise RuntimeError("下载不完整（%d / %d 字节），已丢弃" % (total, size))
            shutil.move(tmp, dst)
            ok.append("已自动补齐 %s：%.0f MB → %s" % (
                name, os.path.getsize(dst) / 1048576.0, _rel(dst)))
        except Exception as exc:
            bad.append("%s 没能自动下载（%s：%s）" % (name, type(exc).__name__, exc))
    return ok, bad


# ── 对外接口：查缺失 / 补权重 / 出一份可读报告 ────────────────────────────────

def check_all():
    need, opt, weights = missing_items()
    return need, list(opt) + [w[0] for w in weights], []


def auto_fetch(_what=None):
    need, _opt, weights = missing_items()
    if not weights:
        return "没有需要补的权重"
    ok, bad = fetch_weights(weights)
    return "；".join(ok + bad) if (ok or bad) else "没有需要补的权重"


def report(only_missing=True):
    need, opt, weights = missing_items()
    L = []
    if not need and not opt and not weights:
        return "环境齐全，可以开工。", True, 0
    if need:
        L.append("★ 缺必需项（不补跑不了）：%s" % "、".join(need))
    if opt:
        L.append("○ 缺可选项（只影响对应功能）：%s" % "、".join(opt))
    if weights:
        L.append("○ 缺权重：%s" % "、".join(w[0] for w in weights))
    L.append(one_click_fix())
    for mod in need + opt:
        for m, purpose, how, _must in PACKAGES:
            if m == mod:
                L.append("  %s —— 管：%s\n      退路（手动）：pip install %s，装完必须重启 ComfyUI"
                         % (mod, purpose, how.split()[-1] if how.startswith("pip") else how))
    for name, rel, url, _size, impact in weights:
        L.append("  %s —— 不装的影响：%s\n      自动：%s\n      手动：下载后放到 models/%s"
                 % (name, impact, "点一次本节点就会自己下", rel.replace("\\", "/")))
        L.append("      地址：%s" % url)
    return "\n".join(L), len(need) == 0, len(need) + len(opt) + len(weights)
