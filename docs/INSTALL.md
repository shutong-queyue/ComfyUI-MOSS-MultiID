# 安装与依赖细节

这份是 [README](README.md) 的附录，只讲"怎么装、谁在下载、装不上怎么办"。
正常安装不需要读它——用 ComfyUI-Manager 装完就能直接跑示例工作流。

---

## 三种装法

| 方式 | 步骤 |
| --- | --- |
| ComfyUI-Manager（推荐） | Manager → Install Custom Nodes 搜 `MOSS-MultiID` → 装完重启 ComfyUI |
| git | `cd ComfyUI/custom_nodes && git clone <仓库地址> ComfyUI-MOSS-MultiID`，重启后在 Manager 里对本插件点一次 **Reinstall** |
| 下载 zip | 解压到 `ComfyUI/custom_nodes/ComfyUI-MOSS-MultiID`，然后**双击文件夹里的 `install.bat`**，重启 ComfyUI |

Manager 安装时会自动做两件事：先按 `requirements.txt` 逐行装 Python 包，再自动运行 `install.py` 补权重。全程不需要你敲命令。

## 谁负责下载什么

| 东西 | 谁下 | 大小 | 落在哪 |
| --- | --- | --- | --- |
| insightface / onnxruntime / opencv-python | Manager 读 `requirements.txt`，或 `install.bat` | 见下方备注 | Python 的 `site-packages` |
| sam2 这个库（自动逐人切要用） | `install.py` 自动补 | 约 1 MB | `site-packages` |
| SAM2 权重 `sam2.1_hiera_base_plus.pt` | `install.py` 自动补；漏了的话节点第一次用也会自己下 | 309 MB | `ComfyUI/models/SAM2/` |
| 人脸模型 antelopev2（5 个 onnx） | **insightface 库**第一次被调用时自己下 | 下载 344 MB / 解压后 408 MB | `ComfyUI/models/insightface/models/antelopev2/` |

备注：

- **人脸模型不是本插件下的**，是 insightface 库自带的行为（`insightface/utils/storage.py` 里的 `ensure_available`：
  目标目录不存在就去官方 release 页拉 `antelopev2.zip` 并解压）。本插件重复下一个只会多占空间、还可能和库认的版本对不上，
  所以 `install.py` 不管它。第一次点「人脸定位地盘」多等的那一两分钟就是它在下载，不是卡死。
- Python 包的实际体积很不稳定：opencv 约 123 MB、onnxruntime 取决于装的是 CPU 版还是 GPU 版（GPU 版本机 387 MB）、
  insightface 约 2 MB，再加连带的 scipy / scikit-learn / matplotlib 约 150 MB（后三样任何整合包基本都已经有了）。
  全新环境按最保守估计也要 130 MB 起。
- `sam2` 库故意不放进 `requirements.txt`：那个文件是逐行 pip，任何一行失败 Manager 会把整个插件判成安装失败，
  而本插件没有它照样能跑（只是少了"自动逐人切"这一档）。所以它放在 `install.py` 里单独补，装不上只报告、不牵连别的。

## 离线 / 内网机器

把下面三个文件按路径放好，全程可以不联网：

```
ComfyUI/models/SAM2/sam2.1_hiera_base_plus.pt
ComfyUI/models/insightface/models/antelopev2/scrfd_10g_bnkps.onnx
ComfyUI/models/insightface/models/antelopev2/glintr100.onnx
```

- 本插件只用得上 antelopev2 里的这两个 onnx（检脸 ＋ 比对）。**只要那个目录已经存在，库就不会再去下整包 zip**，
  能省 140 MB 左右。
- SAM2 官方权重地址：`https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt`
  （只有 2.1 这一档可用，1.0 的地址已失效）。
- Python 包用 `pip download` / `pip install --no-index --find-links` 那套常规做法，不在此赘述。

## 装完怎么确认

1. 重启 ComfyUI（装了 Python 包不重启一定不生效）。
2. 拖入 `examples/` 里任意一份示例工作流，直接排队跑。
3. 看「人脸定位地盘」的说明区：**没有任何 `【环境】` 开头的行就是齐了**。
   出现 `【环境】` 行就是还缺东西，那句话里会直接写缺什么、怎么补。

本插件**没有"环境体检"节点**：缺依赖由「人脸定位地盘」开工时静默自查兜底，缺权重它自己下，补不动才说一句。

## 常见问题

**说明里写"草稿图里一张脸都没检到"**
草稿里没有朝向镜头的脸（全侧、全低头、被挡住）。换一张脸看得清的草稿；或者把『遮罩来源』改成『不接遮罩』，
在『备用方框』里给出位置，让它按方框退让。

**出图尺寸怪、最后几条边没变化**
画布宽/高请取 16 的倍数。不是 16 的倍数时最后几行/列不属于任何人，说明区会警告并给出建议尺寸。

**第一次跑特别慢**
多半是 insightface 正在下人脸模型（344 MB）。第二次开始就正常了。

**想要自己的 SAM3**
「人脸定位地盘」的『SAM模型』下拉会列出本机所有 SAM 权重并标注能不能直接用。
标了『需外接节点』的（例如 SAM3）本插件不调用，说明区会给出确切的两步接法。

**改了代码不生效**
必须重启 ComfyUI 进程，刷新页面没用。

## 许可

| 东西 | 许可 | 商用 |
| --- | --- | --- |
| 本插件代码 | MIT | 可以 |
| insightface 库的代码 | MIT | 可以 |
| **insightface 提供的人脸模型（含自动下载的 antelopev2）** | 官方写明 *non-commercial research purposes only* | **不可以**，仅限非商业研究 |
| SAM2 代码与官方权重 | Apache-2.0 | 可以 |

也就是说：**插件本身可以商用，但"检脸/比对"这一步用到的人脸模型不行。**
要商用的话，请自行确认底模、人物滤镜（LoRA）以及这条人脸链路的许可边界。
清单里其它第三方模型（SAM3、sam_hq 等）请按各自发布页自行核对，本插件不对第三方模型的许可作担保。
