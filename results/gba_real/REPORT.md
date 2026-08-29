# AlphaEarth 大湾区真实数据复现报告

## 结论

大湾区真实数据支持“AlphaEarth 嵌入包含可解释的地表信息”，但不支持把原论文在美国本土得到的强预测力和极小空间泛化损失直接外推到大湾区。严格口径主结果中，随机森林平均随机交叉验证 R² 为 **0.662**，仅 **1/26** 个变量超过 0.9、**17/26** 个超过 0.7；前十变量平均空间降幅为 **0.291**。时间稳定性为 **0.869**，Spearman 与随机森林重要性的收敛度为 **0.215**。

## 数据与单条观测面积

- 区域：粤港澳大湾区 9 个广东城市加香港、澳门，GADM 边界面积约 55,470 km²。
- 时间：2017–2023，共 7 年。
- 网格：0.025°，每年 7,802 个有效位置，共 54,614 行；大湾区纬度下相邻网格约覆盖 7.05–7.20 km²。
- AlphaEarth：官方 Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`，64 维；按论文方法以 1 km scale、500 m 半径圆形缓冲区提取。单条嵌入观测的缓冲面积为 **0.7854 km²**。
- 原生 AlphaEarth 像元：10 m × 10 m，即 100 m²。单个官方 COG 文件为 81.92 km × 81.92 km，即约 6,710.89 km²；文件覆盖面积不是单条观测面积。
- 环境变量：26 项。各连续数据源先以双线性方式统一到 30 m 投影，再按论文标注的 10 m extraction scale 取点值；53,235/54,614 行（97.48%）26 项完整。四项土壤变量各缺失 2.42%，主要是近岸覆盖空缺；未插补。

## 与原论文的关键对比

| 指标 | 原论文 CONUS | 大湾区严格复现 | 差异 |
|---|---:|---:|---:|
| 样本量 | 约 12.1M | 54,614 | -99.55% |
| Spearman 每维最大 |ρ| > 0.5 | 34/64 | 33/64 | -1 |
| Spearman 每维最大 |ρ| > 0.7 | 6/64 | 14/64 | +8 |
| RF R² > 0.9 | 12/26 | 1/26 | -11 |
| RF R² > 0.7 | 20/26 | 17/26 | -3 |
| RF 前十变量平均空间 ΔR² | 0.009 | 0.291 | +0.282 |
| 平均时间稳定性 r | 0.963 | 0.869 | -0.094 |
| 稳定性 r > 0.95 | 51/64 | 20/64 | -31 |
| 稳定性 r > 0.90 | 59/64 | 39/64 | -20 |
| |ρ| 与 RF 重要性相关 | 0.450 | 0.215 | -0.235 |

原论文还报告 Transformer 全部 26 项平均空间 ΔR²=0.017；本次没有复现 Transformer 和后续 RAG/LLM 评测，因此不能把 RF 结果冒充 Transformer 对比。

## 最强单维 Spearman 关系

| 环境变量 | 嵌入维度 | ρ |
|---|---:|---:|
| Elevation | A08 | 0.869 |
| Tree cover | A08 | 0.794 |
| Built-up fraction | A08 | -0.793 |
| Slope | A51 | -0.719 |
| Population density | A34 | 0.675 |
| EVI mean | A34 | -0.655 |
| Air temperature | A04 | -0.644 |
| NDVI mean | A34 | -0.616 |

## 随机森林前十变量

| 变量 | 随机 CV R² | 0.5° 空间 CV R² | ΔR² |
|---|---:|---:|---:|
| Elevation | 0.905 | 0.866 | 0.039 |
| EVI mean | 0.826 | 0.451 | 0.375 |
| NDVI mean | 0.817 | 0.385 | 0.432 |
| Air temperature | 0.816 | 0.594 | 0.222 |
| Built-up fraction | 0.813 | 0.761 | 0.051 |
| Dew point | 0.799 | 0.580 | 0.219 |
| NDVI max | 0.796 | 0.276 | 0.520 |
| LST daytime | 0.788 | 0.487 | 0.301 |
| LAI mean | 0.785 | 0.216 | 0.568 |
| Annual ET | 0.762 | 0.578 | 0.184 |

## 空间尺度敏感性

0.5° 块在大湾区形成 37 个空间组，五折样本比例均约 20%。原论文的 2° 块在大湾区只形成 5 个严重不均衡的组，最小折仅 42 行，因此不适用。0.25°、0.5°、1° 敏感性结果显示：海拔、建成区和年降水在更大块下仍保持正 R²；NDVI、LAI、夜间灯光和人口密度随块增大显著下降，1° 时部分变量为负，说明这些关系主要是局地可迁移，而非区域外推稳健。

## 不能归因于“区域”的方法差异

1. 原论文研究区是美国本土，样本量约为本次的 222 倍。
2. PRISM 和 NLCD 不覆盖中国。本次以 ERA5-Land 替代 PRISM 的气温、露点和降水，以 ESA WorldCover 建成区比例替代 NLCD impervious；这些变量不是同源复现。
3. 原论文未完整公开 RF 超参数。本次为 50 棵树、`min_samples_leaf=2`、`max_features=1.0`，每变量最多 30,000 行、5 折。
4. 空间块从论文 2° 改为适合区域范围的 0.5°，并提供 0.25°/1° 敏感性。
5. 为避免粗分辨率数据在小区域统计时只命中像元中心，环境影像内部先统一到 30 m 投影，再以 10 m sampling scale 取点；这与论文仅写“resampled to 10 m”并非逐行代码级完全相同。
6. 未复现多任务 Transformer、FAISS/RAG 与 LLM-as-Judge；报告只对已真实重算的 Spearman、RF、空间验证和时间稳定性负责。

## 文件

- `analysis_results_paper_method.json`：最终完整数值。
- `rf_scores_paper_method.csv`：26 个变量逐项 RF 指标。
- `spatial_sensitivity_paper_method.json`：三种块尺度及每折得分。
- `rf_random_vs_spatial.png`：随机与空间 CV 对比。
- `spearman_heatmap.png`：64×26 Spearman 矩阵。
- `spatial_block_sensitivity.png`：块尺度敏感性。

## 数据来源

- AlphaEarth 官方目录与 Earth Engine：<https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL>
- AlphaEarth GCS/COG 文件结构：<https://developers.google.com/earth-engine/guides/aef_on_gcs_readme>
- ERA5-Land：<https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_MONTHLY_AGGR>
- GADM 中国行政边界：<https://gadm.org/download_country.html>
- 原论文：Rahman (2026), arXiv:2602.10354。
