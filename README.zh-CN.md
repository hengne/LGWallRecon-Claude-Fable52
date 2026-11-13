# LGWallRecon

**由入口面扫描图反演光导内壁形貌的工具包。**
[English](README.md) · [用户手册 (中文)](docs/manual/LGWallRecon_manual_zh.pdf) · [User manual (EN)](docs/manual/LGWallRecon_manual_en.pdf)

作者：李衡讷（Hengne Li），华南师范大学 — <Hengne.Li@cern.ch>，<Hengne.Li@m.scnu.edu.cn>

一根锥形空心光导（入口 ⌀17.5 mm，出口 ⌀8 mm，长 100 mm，内壁贴反射膜），用小的漫射光斑在入口面上逐点扫描，
出口端用 PMT 读出。理想锥面的响应几乎是均匀的（±0.2% 以内），而实测响应图上有百分之几的结构。LGWallRecon 做四件事：

1. 用镜面光线追迹模拟光导，壁面 = CAD 锥面 + 一张自由形状的 B 样条偏差场（CPU 用 numba，GPU 用 OpenCL，每秒数千万条光线）；
2. 由 100×100 点实测图反演出一块能复现它的壁面；
3. 用 Geant4、互易性恒等式、闭合测试以及对同一张扫描图的另一次拟合来检验自己；
4. 输出可缩放的矢量图，以及带三维光线动画的单文件报告网页。

![实测、模拟与残差](results/zh/response_maps.svg)

## 一段话的结论

可以找到一块壁面，使模拟与实测在孔径内区只差 0.90%（均方根），单点测量噪声约 0.8%（χ²/N = 1.27）；理想锥面则相差 5.4%。
这块壁面的局部起伏很小，均方根只有 13 µm，叠加在内腔整体比 CAD 细 0.1–0.9 mm 的轴对称轮廓上。
**但这块壁面不是唯一解**，有两项独立的证据。
（1）用已知壁面做闭合测试时，假扫描图被拟合到噪声水平，恢复出的壁面与真实壁面的相关系数却只有 0.18。
只有在扫描图定得最准的约 100 种形状组合内恢复得好（相关系数 0.9 以上），而这些组合只占真实形变的约 2%；
从“定得准”到“定不准”是渐变的。
（2）此前发布的那块壁面（在手册所列的追迹器修正之前得到），用现在的追迹器重算仍能把实测图复现到 0.92%，
但它的局部起伏与现在这块壁面的相关系数只有 0.15（入口后 20 mm 内为 0.80，再往里不相关），两条平均轮廓相差均方根 0.2 mm。
所以壁面图应读作“能解释扫描图的一个正则化解”：它说明很小的起伏就够了，而不是真实内表面的样子。
准确数字见 `results/numbers.json`。

![内壁形貌图](results/zh/inner_surface.svg)

## 快速开始

```bash
git clone git@github.com:hengne/LGWallRecon-Claude-Fable52.git && cd LGWallRecon-Claude-Fable52
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -r requirements-gpu.txt    # 可选：OpenCL 追迹器
./run_all.sh                  # 自检 -> 量孔径 -> 反演 -> 出图和报告（有 GPU 时约 45 分钟）
./run_all.sh --with-checks    # 另外跑闭合测试、互易性、放开反射率和 Geant4 复算（再加约 2 小时）
```

输出在 `results/`（`zh/`、`en/` 两套图，`wall_delta.csv`、`wall_coeffs.json`、`numbers.json`）和
`docs/LGWallRecon_report_{zh,en}.html`（用浏览器打开，离线可用）。

环境要求：Python ≥ 3.10 及 `requirements.txt` 中的包。可选：支持 OpenCL 的 GPU 及 `requirements-gpu.txt` 中的包
（`clinfo` 能列出设备；没有就设 `LG_GPU=0`，慢约二十倍）、Geant4 11.x（交叉验证用）、`node`（生成光线动画 SVG 用）。
所有作业通过 `tools/run_limited.sh` 启动：用 systemd scope 设内存硬上限，并按机器空闲程度分配 CPU；
没有 `systemd-run` 的系统上直接调用 Python 脚本即可。任何一步或自检失败，`run_all.sh` 都会停下；
没有针对当前壁面重做的检验，报告里会标明。

## 分步说明

以下命令都在 `src/` 目录内执行，前面加 `../tools/run_limited.sh`（下表简写为 `R`）。

| 步骤 | 命令 | 作用 |
|---|---|---|
| 1 | `R python edge.py`、`R python edge_spot.py` | 从扫描图的锐利边缘量出入口中心、半径、椭圆度和光斑形状 → `out/edge.npz` |
| 2 | `R ./pipeline.sh data` | 下面的整条反演链 → `out/fitB_datab.npz`、`out/final_data.npz` |
| 2a | `stageA.py 10 fix 3.0` | 由径向响应求轴对称壁面轮廓 |
| 2b | `refine.py`、`kernel_modes.py` | 响应图对每个壁面系数的线性响应（周向傅里叶模式） |
| 2c | `stageB.py` | 二维壁面的阻尼最小二乘反演；每一步都用完整追迹检验 |
| 2d | `final.py` | 高统计模拟图、残差、细网格上的壁面 |
| 3 | `R python compare_walls.py ../out/final_data.npz ../results/earlier/wall_coeffs_cb81f08.json` | 这块壁面与此前对同一张扫描图拟合出的壁面对比 |
| 4 | `R ./closure.sh 1` | 闭合测试与分辨率分析 |
| 4a | `R python test_jacobian.py datab ../out/fitB_datab.npz` | 线性响应核与精确有限变化的对比 |
| 5 | `R python test_reciprocity.py ../out/fitB_datab.npz`、`../g4/run_final.sh ../out/fitB_datab.npz` | 自洽性检验与 Geant4 复算 |
| 6 | `R ./make_report.sh` | 图、数字、壁面数值表、报告网页 |

每个脚本的参数、文件格式、如何改用到别的光导，以及常见问题，见用户手册。

## 检验结果

以下是已发布这次运行的数字（`results/numbers.json`）；`./run_all.sh --with-checks` 会全部重新生成。

| 检验 | 结果 |
|---|---|
| 理想锥面，追迹器 | 面积平均 T = (R₂/R₁)² = 0.2090；逐点从中心 +0.23% 到边缘 −0.13% |
| Geant4，理想锥面 | Geant4 / 追迹器 = 1.0012 ± 0.0011 |
| Geant4，反演壁面（72 万面片） | Geant4 / 追迹器 = 1.0007 ± 0.0015 |
| GPU 与 CPU 追迹器 | 双精度下逐点透过率相同（只差舍入误差）；单精度平均差 (−2 ± 3)×10⁻⁶ |
| 互易性（正向光通量 = 反向光通量） | 比值 1.003 ± 0.009 |
| 准随机采样、掠射撞壁 | 自检，`run_all.sh` 最先执行 |
| 线性响应核与精确变化的对比 | 非轴对称平滑形变：相关系数 0.84–0.99；轴对称形变预言得差（平缓鼓包 0.44–0.50），见 `results/checks/` |
| 闭合测试：响应图 | 复现到噪声水平（χ²/N = 0.95） |
| 闭合测试：壁面 | **未能恢复**：相关系数 0.18；在定得最准的前 30 / 100 / 150 种形状组合内为 0.97 / 0.91 / 0.70 |
| 同一张实测图的两次拟合 | 都能拟合（0.92% 和 0.90%），**壁面不同**：相关系数 0.15，平均轮廓相差均方根 0.2 mm |
| 放开镜面反射率 | R 走到 0.96，χ² 从 34 降到 27（40 个点）；与轮廓简并，这张扫描图分不开 |

## 目录

```
data/      实测扫描数据（CSV）与 CAD 几何（STEP）
src/       追迹器（lgtrace.py CPU，lgcl.py GPU）、反演链、检验、专题研究、出图与报告脚本
g4/        Geant4 交叉验证程序（lgsim.cc）
web/       浏览器端追迹器、报告模板、语言表（i18n/zh.json、i18n/en.json）
tools/     run_limited.sh 资源限制包装脚本
results/   图（zh/、en/）、壁面数值表、数字汇总；earlier/ 里是较早那次拟合的壁面
docs/      单文件报告网页与用户手册
```

## 局限

壁面不唯一（见上），拟合也没有被证明是最优（χ²/N = 1.27；此前对同一张扫描图的拟合停在 1.48 和 1.31）。
周向平均轮廓与信号幅度、出口口径、镜面反射率简并。引导拟合的线性响应核对宽而平滑的形状（首先是平均轮廓）不准，
这些方向上的步长找得很低效，闭合测试的分辨率数字在这里也只能作参考（每一步仍经完整追迹检验）。一切都归因于几何（完美镜面、均匀 PMT、朗伯光源）。
r = 8.0 mm 到入口边沿之间的环带没有拟合，那里实测比模型低约 2%。样条网格为轴向 2.5 mm、周向 15°。
要确定真实壁面，需要增加独立的测量：几个入射角的准直光束扫描、出口端角分辨读出、从出口一侧扫描，或直接做轮廓测量。

## 仓库

GitHub（公开）：<https://github.com/hengne/LGWallRecon-Claude-Fable52> · GitLab（SCNU 组）：<https://gitlab.cern.ch/SCNU/LGWallRecon-Claude-Fable52>

## 许可证

MIT，见 [LICENSE](LICENSE)。

## 引用

Hengne Li（李衡讷），*LGWallRecon: light-guide inner-wall reconstruction from an entrance-face scan*（2026）。
