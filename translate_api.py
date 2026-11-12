"""一句话转构图·API版

四条对齐结果：
  1 模型只输出 0~1 比例坐标，代码换算成任意尺寸像素
  2 工作流里不做深校验，只把模型返回内容原样展示；深校验在我这边脚本里
  3 只做一趟
  4 不用旧规则版兜底；离线走本机 ollama，同样以模型输出为准

翻译技巧（skill）放在同目录 skill.txt，改它不用动代码。
"""

import json
import os
import re
import urllib.request

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(PLUGIN_DIR, "api_config.json")
SKILL_FILE = os.path.join(PLUGIN_DIR, "skill.txt")
SF_URL = "https://api.siliconflow.cn/v1/chat/completions"
SF_LIST = "https://api.siliconflow.cn/v1/models"
OLLAMA = "http://127.0.0.1:11434"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"

FALLBACK_SKILL = ("把一句中文拆成 faces/bodily/props/background，坐标用 0~1 比例，"
                  "只输出 JSON。详细要求请写在插件目录的 skill.txt 里。")


def load_skill():
    try:
        s = open(SKILL_FILE, encoding="utf-8").read().strip()
        if s:
            return s
    except Exception:
        pass
    return FALLBACK_SKILL


def load_cfg():
    try:
        d = json.load(open(CONFIG, encoding="utf-8"))
        return str(d.get("api_key", "")), str(d.get("model", DEFAULT_MODEL))
    except Exception:
        return "", DEFAULT_MODEL


def sf_models(key):
    try:
        req = urllib.request.Request(SF_LIST, headers={"Authorization": "Bearer " + key})
        d = json.loads(urllib.request.urlopen(req, timeout=25).read().decode("utf-8"))
        ids = [str(m.get("id")) for m in (d.get("data") or [])]
        return [i for i in ids if re.search(r"(DeepSeek-V4|DeepSeek-V3\.2|Qwen3\.5|GLM-5)", i)
                and "Pro/" not in i and not any(x in i for x in
                    ("OCR", "Embedding", "Reranker", "ASR", "VL", "Omni", "Coder"))][:20]
    except Exception:
        return []


def ollama_models():
    try:
        d = json.loads(urllib.request.urlopen(OLLAMA + "/api/tags", timeout=5).read().decode("utf-8"))
        return ["本机/" + m["name"] for m in d.get("models", [])]
    except Exception:
        return []


def to_pixels(box, W, H):
    try:
        x0, y0, x1, y1 = [float(v) for v in box]
    except Exception:
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return [int(y0 * H - H / 2), int(x0 * W - W / 2),
            int(y1 * H - H / 2), int(x1 * W - W / 2)]


def to_point(v, W, H):
    """把模型给的脸位置统一成一个点（绝对像素）。
    两个数 = 直接就是点；四个数 = 取框中心。数值明显超出 0~1 的当作它给了像素，不收。
    """
    if v is None:
        return None
    try:
        nums = [float(x) for x in v]
    except Exception:
        return None
    if len(nums) == 2:
        x, y = nums
    elif len(nums) >= 4:
        x = (nums[0] + nums[2]) / 2.0
        y = (nums[1] + nums[3]) / 2.0
    else:
        return None
    if max(abs(x), abs(y)) > 1.6:          # 明显不是比例，别把 1440 当成 1.44
        return None
    x = min(max(x, -0.05), 1.05)
    y = min(max(y, -0.05), 1.05)
    return int(round(x * W)), int(round(y * H))


PLACEHOLDER = ("未提及", "原话没有说明", "原话没提", "自己的发型发色", "不知道", "N/A", "无（")


class MOSSIdTranslate:
    @classmethod
    def INPUT_TYPES(cls):
        key, saved = load_cfg()
        models = sf_models(key) if key else []
        models += ollama_models()
        if not models:
            models = [DEFAULT_MODEL]
        default = saved if saved in models else (DEFAULT_MODEL if DEFAULT_MODEL in models else models[0])
        return {"required": {
            "这句话": ("STRING", {"multiline": True, "default":
                "两名年轻女子并排站在操场边看着镜头，左边的人穿粉色背心，右边的人穿黄色夹克外套，"
                "两人都微笑，半身构图，白天明亮自然光，写实照片",
                "tooltip": "要生成的画面描述，自然语言即可。本节点只把它翻译成带坐标的结构，"
                           "不会替你重新构思画面，也不建议用它决定最终构图"}),
            "已知角色名单": ("STRING", {"default": "角色一,角色二",
                "tooltip": "只是对照表，不是人数要求。原话点到谁就用谁的名字；"
                           "原话里有但名单没有的人，会沿用原话里的称呼"}),
            "模型": (models, {"default": default,
                "tooltip": "以「本机/」开头的走本地离线模型，其余走在线接口；"
                           "在线接口需要在插件文件夹里的 api_config.json 填密钥"}),
            "画布宽": ("INT", {"default": 1440, "min": 64, "max": 8192, "step": 16,
                "tooltip": "画面宽度（像素）。坐标会按它换算成格子，必须和实际出图尺寸一致；建议取 16 的倍数"}),
            "画布高": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 16,
                "tooltip": "画面高度（像素）。同上，要和实际出图尺寸一致"}),
            "翻译技巧": ("STRING", {"multiline": True, "default": load_skill(),
                "tooltip": "翻译时遵循的规则文本，默认读插件文件夹里的 skill.txt；"
                           "也可以由『构图预设管理』送进来一份覆盖它"}),
            "随机性": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05,
                "tooltip": "随机性越低，同样的输入每次结果越接近；越高越灵活但越不稳定"}),
            },
            "optional": {"技巧覆盖": ("STRING", {"forceInput": True,
                "tooltip": "可选：接一份规则文本进来，优先于上面的『翻译技巧』"})}}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("构图指令", "地盘方框", "负向提示词", "翻译结果")
    FUNCTION = "run"
    CATEGORY = "MOSS多身份"
    DESCRIPTION = "把一句中文拆成带比例坐标的构图 JSON，代码换算像素，只出一趟"

    def run(self, 这句话, 已知角色名单, 模型, 画布宽, 画布高, 翻译技巧, 随机性, 技巧覆盖=None):
        names = [s.strip() for s in str(已知角色名单).replace("，", ",").split(",") if s.strip()]
        W, H = int(画布宽), int(画布高)
        skill = str(技巧覆盖) if 技巧覆盖 and str(技巧覆盖).strip() else str(翻译技巧)
        user = "已知角色名单（仅供对照，不代表人数）：%s\n原话：%s" % (
            "、".join(names) if names else "（无）", 这句话)
        key, _ = load_cfg()
        if str(模型).startswith("本机/"):
            src = "本机 ollama（离线）"
            body = {"model": str(模型).replace("本机/", ""), "system": skill, "prompt": user,
                    "format": "json", "stream": False,
                    "options": {"temperature": float(随机性), "num_predict": 2000}}
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                OLLAMA + "/api/generate", data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"}), timeout=400).read().decode("utf-8"))
            raw, tokens = r.get("response", ""), 0
        else:
            if not key:
                raise RuntimeError("没有硅流密钥：写进插件目录 api_config.json 的 api_key，"
                                   "或把「模型」换成以 本机/ 开头的离线模型。")
            src = "硅流 API"
            body = {"model": 模型, "max_tokens": 2000, "temperature": float(随机性),
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": skill},
                                 {"role": "user", "content": user}]}
            r = json.loads(urllib.request.urlopen(urllib.request.Request(
                SF_URL, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + key}),
                timeout=180).read().decode("utf-8"))
            raw = r["choices"][0]["message"]["content"]
            tokens = (r.get("usage") or {}).get("total_tokens", 0)
        try:
            d = json.loads(raw)
        except Exception:
            raise RuntimeError("%s 返回的内容不是合法 JSON，原文如下：\n%s" % (src, str(raw)[:900]))

        faces = d.get("faces") or []

        bodily = d.get("bodily") or []
        props = d.get("props") or []
        view = str(d.get("view", "平视")).strip()
        bgtext = str(d.get("background", "")).strip()
        extra = d.get("extra_faces", 0)
        items, region, warn = [], [], []

        for i, f in enumerate(faces[:6]):
            nm = str(f.get("name", "")).strip()
            look = str(f.get("look", "")).strip().replace("\n", "，")
            pose = str(f.get("pose", "")).strip().replace("\n", "，")
            for ph in PLACEHOLDER:
                if ph in look:
                    warn.append("look 里有占位废话「%s」（skill 要补一条禁止它）" % ph)
                    look = ""
                    break
            desc = "，".join([x for x in (look, pose) if x])
            pb = to_pixels(f.get("box"), W, H)
            if pb is None:
                seg = 1.0 / max(1, len(faces))
                pb = to_pixels([i * seg, 0.10, (i + 1) * seg, 0.94], W, H)
                warn.append("%s：坐标缺失或非法，已按人数均分补上" % (nm or i + 1))
            cap = ("%s，%s。" % (nm, desc)) if desc else ("%s。" % (nm or "人物%d" % (i + 1)))
            items.append({"label": "character", "caption": cap, "bbox": pb})
            # 脸位点：模型给两个数就直接用；给四个数就取中心；都没给才按身体框估一个
            pt = to_point(f.get("face") or f.get("face_point") or f.get("head") or f.get("face_box"), W, H)
            bx0, by0, bx1, by1 = pb[1] + W / 2, pb[0] + H / 2, pb[3] + W / 2, pb[2] + H / 2
            if pt is None:
                hx, hy = int((bx0 + bx1) / 2), int(by0)
                src = "按身体框顶部估的"
                warn.append("「%s」模型没交脸位点，下面这个点是按身体框顶部估的（不能当作模型点准确的证据）"
                            % (nm or (i + 1)))
            else:
                hx, hy = pt
                src = "模型给的"
                if not (bx0 - 0.08 * (bx1 - bx0) <= hx <= bx1 + 0.08 * (bx1 - bx0)):
                    warn.append("「%s」的脸位点 x%d 跑到自己身体框（x%d~%d）外面了，交界会不稳"
                                % (nm or (i + 1), hx, bx0, bx1))
            region.append({"owner": i + 1, "name": nm or ("%d号" % (i + 1)),
                           "x": int(bx0), "y": int(by0),
                           "w": int(bx1 - bx0), "h": int(by1 - by0),
                           "hx": hx, "hy": hy, "head_src": src})

        for b in bodily:
            what = str(b.get("what", "")).strip().replace("\n", "，")
            if not what:
                continue
            for nm in names:
                what = what.replace(nm, "被点名角色")
            pb = to_pixels(b.get("box"), W, H) or to_pixels([0.05, 0.78, 0.95, 1.0], W, H)
            items.append({"label": "object",
                          "caption": "%s（属于%s，此处没有脸，不许长出脸）"
                                     % (what, str(b.get("owner", "拍摄者"))),
                          "bbox": pb})

        for pr in props[:8]:
            what = str(pr.get("what", "")).strip().replace("\n", "，")
            pb = to_pixels(pr.get("box"), W, H) if what else None
            if pb:
                items.append({"label": "object", "caption": what, "bbox": pb})

        if bgtext:
            items.append({"label": "object",
                          "caption": "背景远景：%s，占画面顶部一条横向区域，"
                                     "背景里的人必须模糊、不可辨认五官" % bgtext[:120],
                          "bbox": to_pixels([0.0, 0.0, 1.0, 0.22], W, H)})

        nface = len([i for i in items if i["label"] == "character"])
        layout = {"summary": {"meta": ["photo", "two people" if nface == 2 else "multiple people"],
                              "canvas": "single",
                              "caption": "%s。画面里清晰可辨的人脸共 %d 张，每张脸各自独立清晰，"
                                         "除此之外不得出现任何可辨认的多余人脸。" % (view, nface)},
                  "items": items, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}

        negative = ("nsfw, worst quality, bad anatomy, watermark, jpeg artifact, third person, "
                    "extra person, extra face, additional head, duplicated face")
        if int(extra or 0) > 0:
            negative += ", blurred crowd only"

        lines = ["来源：%s 模型：%s%s" % (src, 模型, ("  tokens：%d" % tokens) if tokens else ""),
                 "视角：%s" % view,
                 "有脸主体 %d 个，无脸部件 %d 条，物件 %d 条，背景可辨认人数 %s"
                 % (nface, len(bodily), len(props), extra), ""]
        for it in items:
            lines.append("[%s] x%d~%d y%d~%d  %s" % (
                it["label"], int(it["bbox"][1] + W / 2), int(it["bbox"][3] + W / 2),
                int(it["bbox"][0] + H / 2), int(it["bbox"][2] + H / 2), it["caption"][:56]))
        for r in region:
            lines.append("脸位点（%s）%s号 %s：x%d y%d ｜ 她身体框内头顶预留 y%d 起" % (
                r.get("head_src", "?"), r["owner"], r["name"], r.get("hx", 0), r.get("hy", 0), r["y"]))
        if len(region) >= 2:
            hs = [(r.get("hx", 0), r.get("hy", 0)) for r in region]
            dd = min((((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
                     for i, a in enumerate(hs) for b in hs[i + 1:])
            lines.append("两个人脸位点相距：%d 像素（保护范围按这个的一半画，太小就会互相挤）" % dd)
        if len(region) >= 2:
            a, b2 = region[0], region[1]
            gap = b2["x"] - (a["x"] + a["w"])
            lines.append("人物框之间的空档：%d 像素（应为 0，否则模型会往里塞人）" % gap)
        if warn:
            lines.append("")
            lines += ["⚠ " + w for w in warn]
        lines += ["", "模型原始返回（照抄，方便截图反馈）：", str(raw)[:1600]]
        return (json.dumps(layout, ensure_ascii=False, separators=(",", ":")),
                json.dumps(region, ensure_ascii=False), negative, "\n".join(lines))


NODE_CLASS_MAPPINGS = {"MOSSIdTranslate": MOSSIdTranslate}
NODE_DISPLAY_NAME_MAPPINGS = {"MOSSIdTranslate": "一句话转构图·API版"}
