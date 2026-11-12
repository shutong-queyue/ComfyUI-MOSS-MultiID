"""一键补齐：把本插件需要的权重放到正确位置（ComfyUI-Manager 会自动运行本脚本）

单独运行时：
    python install.py            只补缺的权重（已就位就跳过，不重复下载）
    python install.py --check    只查不下载，打印缺什么
    python install.py --force    已存在也重新下载

Python 包主要由 requirements.txt 负责（Manager 会先逐行装它）。
只有一个例外：自动切遮罩用的 sam2 库在这里补——它装失败不该连累整个插件被判安装失败。
人脸模型（antelopev2）也不在这里下，insightface 库第一次用会自己去下。
"""
import argparse
import os
import shutil
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def comfy_root():
    """从插件文件夹往上找 ComfyUI 根目录：认两个特征——有 custom_nodes、有 models。
    只按固定层数算会错一层（这里就错过一次），所以改成逐级探测。"""
    p = HERE
    for _ in range(6):
        p = os.path.dirname(p)
        if os.path.isdir(os.path.join(p, "custom_nodes")) and os.path.isdir(os.path.join(p, "models")):
            return p
    # 兜底：custom_nodes 的上一级
    return os.path.dirname(os.path.dirname(HERE))


MODELS = os.path.join(comfy_root(), "models")

# (说明, 判定用的文件, 下载 URL, 落地方式, 期望字节数)
#
# 注意：人脸模型 antelopev2 不在这里下。
# insightface 这个库第一次用的时候会自己去 GitHub 的 release 页把
# antelopev2.zip 拉下来解压到 models/insightface/models/antelopev2/，
# 这是库自带的行为（见 insightface/utils/storage.py），很多插件都靠它。
# 我们重复下一个反而多占 300MB、还可能和库的版本对不上，所以不管它。
ITEMS = [
    ("SAM2 检查点（每人一块自动切要用）",
     os.path.join(MODELS, "SAM2", "sam2.1_hiera_base_plus.pt"),
     "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
     "file", 323606802),
]


def human(n):
    return "%.0f MB" % (n / 1048576.0) if n >= 1048576 else "%.0f KB" % (n / 1024.0)


def download(url, dst, expect=None):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "comfyui-moss-multiid"})
    total = 0
    with urllib.request.urlopen(req, timeout=900) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            total += len(chunk)
            done = total * 100 // expect if expect else 0
            sys.stdout.write("\r      下载中 %s / %s  %d%%   " % (
                human(total), human(expect) if expect else "?", done))
            sys.stdout.flush()
    print()
    if expect and total < expect * 0.98:
        os.remove(tmp)
        raise RuntimeError("下载不完整，已丢弃")
    return tmp


def sam2_installed():
    """只查有没有这个包，不真的 import（它一进来就会拉起 torch，慢）。"""
    import importlib.util
    try:
        return importlib.util.find_spec("sam2") is not None
    except Exception:
        return False


def ensure_sam2_package():
    """「每人一块（自动切）」用的 sam2 库。

    它不在 requirements.txt 里，因为那文件是逐行 pip，一行挂掉会让 Manager
    把整个插件判成安装失败——而本插件不装它也能跑（只是少了自动切这一档）。
    所以放到这里单独补：装不上只报告，不牵连别的。
    """
    if sam2_installed():
        print("  ✓ sam2 库已在（自动切遮罩可用）")
        return True
    print("  ○ sam2 库不在，正在安装（约 1～2 分钟）……")
    import subprocess
    cmd = [sys.executable, "-m", "pip", "install",
           "git+https://github.com/facebookresearch/sam2.git"]
    try:
        r = subprocess.run(cmd, timeout=1800)
        if r.returncode == 0:
            print("  ✓ sam2 库装好了")
            return True
        print("  ! sam2 库没装上（退出码 %d）。自动切遮罩这一档暂时用不了，"
              "其余功能不受影响；可以稍后重试或手动执行：" % r.returncode)
        print("      %s" % " ".join(cmd))
    except Exception as exc:
        print("  ! sam2 库安装出错：%s。自动切遮罩这一档暂时用不了，其余功能不受影响。" % exc)
    return False


def main():
    ap = argparse.ArgumentParser(description="补齐 MOSS 多身份插件需要的权重")
    ap.add_argument("--check", action="store_true", help="只检查不下载")
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()

    print("模型根目录：%s" % MODELS)
    print("说明：人脸模型（antelopev2）由 insightface 库首次使用时自动下载，本脚本不管。")
    todo, missing = [], []
    for name, target, url, how, expect in ITEMS:
        have = os.path.exists(target) and not args.force
        print("  %s %s" % ("✓" if have else "○", name))
        print("      位置：%s" % os.path.relpath(target, MODELS))
        if have:
            print("      已就位（%s）" % human(os.path.getsize(target)))
        else:
            todo.append((name, target, url, how, expect))
            missing.append(name)

    if args.check:
        lib = sam2_installed()
        print("  %s sam2 库（自动切遮罩用）" % ("✓" if lib else "○"))
        bad = list(missing) + ([] if lib else ["sam2 库"])
        print("\n检查结果：%s" % ("全部就位，可以开工" if not bad else "还缺 %d 项：%s" % (
            len(bad), "、".join(bad))))
        return 0 if not bad else 1

    if not todo:
        print("\n权重都已就位，不重复下载。")
        print("\n检查自动切遮罩用的库：")
        ensure_sam2_package()
        return 0

    print("\n开始补齐 %d 项（约 309MB，只在第一次发生）：" % len(todo))
    failed = []
    for name, target, url, how, expect in todo:
        print("  → %s" % name)
        try:
            tmp = download(url, target, expect)
            shutil.move(tmp, target)
            print("      已保存 %s（%s）" % (os.path.relpath(target, MODELS), human(os.path.getsize(target))))
        except Exception as exc:
            failed.append(name)
            print("      失败：%s（可稍后手动下载：%s）" % (exc, url))

    print("\n完成：%d 项成功，%d 项失败。" % (len(todo) - len(failed), len(failed)))
    if failed:
        print("失败的项：%s" % "、".join(failed))
        return 1
    print("\n检查自动切遮罩用的库：")
    ensure_sam2_package()
    print("\n权重齐了。Python 包由 requirements.txt 负责（Manager 会先装它）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
