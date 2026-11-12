"""构图预设管理 + 出图核对

两个节点：
  构图预设管理：把发给大模型的那段要求存成多套预设，可以列出/读取/保存/删除，
                存到插件自己的 presets 文件夹里，不碰 ComfyUI 本体。
  出图核对：    把成品图和你的原话一起交给本机视觉大模型，逐条检查动作、视角、
                服装有没有照做，输出中文核对清单和遵从度分数。
"""

import base64
import io
import json
import os
import re
import urllib.request

import numpy as np
import torch

OLLAMA = "http://127.0.0.1:11434"
PRESET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")


def _ensure_dir():
    if not os.path.isdir(PRESET_DIR):
        os.makedirs(PRESET_DIR, exist_ok=True)
    return PRESET_DIR


def _safe(name):
    s = re.sub(r"[^0-9A-Za-z_\u4e00-\u9fff-]", "", str(name))
    return s[:40] or "未命名"


def _list_presets():
    _ensure_dir()
    out = []
    for f in sorted(os.listdir(PRESET_DIR)):
        if f.endswith(".json"):
            try:
                d = json.load(open(os.path.join(PRESET_DIR, f), encoding="utf-8"))
                out.append((f[:-5], str(d.get("备注", ""))[:24]))
            except Exception:
                out.append((f[:-5], "（文件读不出来）"))
    return out


def _ollama(text_prompt, images, model, temperature=0.1):
    # num_predict 要给得很宽：带思考段的模型会先长篇推理，
    # 额度小了推理没跑完，正式回答直接是空的
    body = {"model": model, "prompt": text_prompt, "stream": False,
            "options": {"temperature": float(temperature), "num_predict": 8000}}
    if images:
        body["images"] = images
    req = urllib.request.Request(OLLAMA + "/api/generate", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read().decode("utf-8")).get("response", "")


class MOSSIdPresetManager:
    """构图预设管理"""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            names = [n for n, _ in _list_presets()] or ["（还没有存过预设）"]
        except Exception:
            names = ["（读不到预设目录）"]
        return {
            "required": {
                "动作": (["列出所有预设", "读取一个预设", "保存为新预设", "删除一个预设"],
                          {"default": "列出所有预设",
                           "tooltip": "要让这个节点做什么：列出已有预设 / 读取一个 / 保存新的 / 删除一个。"}),
                "预设名": (names, {"tooltip": '从下拉里选一个已有的预设名；配合上面的『动作』决定读哪个、覆盖哪个、删哪个。'}),
                "预设内容": ("STRING", {"multiline": True, "default": "",
                                        "tooltip": "这套要求的完整文本。动作＝保存时，这里写的内容会被存成预设；动作＝读取时，这里会被读到的预设内容覆盖。"}),
                "备注": ("STRING", {"default": "", "multiline": False, "tooltip": '给这套预设留的一句话说明，只在列出和保存时看得到，不参与生成。'}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("预设内容", "操作结果")
    FUNCTION = "run"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "多套构图预设的列出/读取/保存/删除，存在插件自己的 presets 文件夹"

    def run(self, 动作, 预设名, 预设内容, 备注):
        _ensure_dir()
        if 动作 == "列出所有预设":
            lst = _list_presets()
            if not lst:
                return ("", "presets 文件夹是空的。选「保存为新预设」就能存第一套。")
            lines = [f"共 {len(lst)} 套预设（目录：{PRESET_DIR}）"]
            for n, r in lst:
                lines.append(f"  · {n}　{r}")
            return ("", "\n".join(lines))

        if 动作 == "读取一个预设":
            p = os.path.join(PRESET_DIR, _safe(预设名) + ".json")
            if not os.path.exists(p):
                return ("", f"找不到预设「{预设名}」。先选「列出所有预设」看看都有什么。")
            d = json.load(open(p, encoding="utf-8"))
            content = str(d.get("内容", ""))
            return (content, f"已读取预设「{预设名}」（{len(content)} 字）。把「预设内容」输出接到"
                             f"「一句话转构图·大模型版」的「预设覆盖」口即可生效。")

        if 动作 == "保存为新预设":
            if not str(预设内容).strip():
                return ("", "预设内容是空的，没保存。请先把要存的那段要求贴进来。")
            name = _safe(预设名) if 预设名 and 预设名 != "（还没有存过预设）" else "我的预设"
            p = os.path.join(PRESET_DIR, name + ".json")
            json.dump({"名字": name, "备注": 备注, "内容": 预设内容},
                      open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            return (预设内容, f"已保存预设「{name}」→ {p}")

        if 动作 == "删除一个预设":
            p = os.path.join(PRESET_DIR, _safe(预设名) + ".json")
            if not os.path.exists(p):
                return ("", f"找不到预设「{预设名}」，没删除。")
            os.replace(p, p + ".已删除")
            return ("", f"预设「{预设名}」已改名为 {os.path.basename(p)}.已删除（没有真删，可恢复）。")

        return ("", "不认识这个动作")


COLORS = [("黑", (0.05, 0.05, 0.06)), ("白", (0.93, 0.93, 0.92)), ("米白", (0.88, 0.85, 0.78)),
          ("灰", (0.50, 0.50, 0.52)), ("粉红", (0.93, 0.55, 0.66)), ("红", (0.75, 0.12, 0.15)),
          ("酒红", (0.45, 0.10, 0.20)), ("橙", (0.90, 0.50, 0.15)), ("黄", (0.87, 0.78, 0.20)),
          ("绿", (0.20, 0.55, 0.28)), ("深绿", (0.10, 0.32, 0.16)), ("蓝", (0.15, 0.30, 0.75)),
          ("牛仔蓝", (0.30, 0.42, 0.62)), ("紫", (0.45, 0.25, 0.65)), ("棕", (0.45, 0.28, 0.16)),
          ("卡其", (0.62, 0.50, 0.32)), ("银灰", (0.78, 0.78, 0.80))]

VIEW_WORDS = ("俯视", "仰视", "平视", "第一人称", "主观视角", "鸟瞰", "特写", "半身", "全身", "远景")


def _nearest_color(rgb):
    best, name = 9e9, "未知"
    for cn, c in COLORS:
        d = (rgb[0] - c[0]) ** 2 + (rgb[1] - c[1]) ** 2 + (rgb[2] - c[2]) ** 2
        if d < best:
            best, name = d, cn
    return name


def _torso_colors(img, faces):
    """量每个人衣服的主色。

    坑（已踩过）：不能取"脸下方一大片"——那里混着腿、牛仔裤、头发和阴影，
    量出来永远是棕/黑。改成：只取脖子到胸口这一窄条，并且剔掉太暗、太灰的像素，
    剩下的取中位数。这样量到的才是衣服本身。
    """
    H, W = img.shape[0], img.shape[1]
    out = []
    for f in sorted(faces, key=lambda x: x["cx"]):
        fh = max(8.0, f["h"])
        y0 = int(min(H - 2, max(0, f["cy"] + fh * 0.55)))
        y1 = int(min(H, max(y0 + 3, f["cy"] + fh * 1.45)))
        x0 = int(max(0, f["cx"] - fh * 0.75))
        x1 = int(min(W, f["cx"] + fh * 0.75))
        if y1 <= y0 or x1 <= x0:
            continue
        reg = img[y0:y1, x0:x1].reshape(-1, 3)
        mx = reg.max(axis=1)
        sat = mx - reg.min(axis=1)
        keep = reg[(mx > 0.16) & ((sat > 0.05) | (mx > 0.55))]
        if len(keep) < max(20, int(len(reg) * 0.03)):
            keep = reg
        med = np.median(keep, axis=0)
        out.append({"side": "左" if f["cx"] < W / 2 else "右",
                    "颜色": _nearest_color(med), "rgb": [round(float(v), 3) for v in med],
                    "取样区": f"x{x0}~{x1} y{y0}~{y1}（脖子到胸口，已剔掉太暗太灰的像素）",
                    "有效占比": round(float(len(keep) / max(1, len(reg))), 2)})
    return out


NODE_CLASS_MAPPINGS = {"MOSSIdPresetManager": MOSSIdPresetManager}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSIdPresetManager": "构图预设管理"}
