[TOC]

### 模型-经济管理-市场与资产模型-马科维茨的均值方差组合模型【hxy】

#### 1. 模型名称

马科维茨的均值方差组合模型（Markowitz Mean-Variance Model，Markowitz Model，MM)

#### 2. 核心词汇

- 投资组合（Portfolio）
- 收益率（Rate of Return）
- 期望收益率（Expected Rate of Return）
- 方差（Variance）
- 系统性风险（Systematic Risk）
- 非系统性风险（Unsystematic Risk）

#### 3. 基本内容

##### 3.1 假设

- 市场中存在$N \geq 2$个风险资产
- 投资者是风险规避的，在收益相等的情况下投资者会选择风险最低的投资组合
- 投资期限为一期，在期初时投资者按照效用最大化的原则进行资产组合的选择
- 市场是完善的，无交易成本，而且风险资产可以无限细分
- 投资者在最优资产组合的选择过程中，只关心风险资产的均值、方差以及不同资产间的协方差

##### 3.2 模型建立

- 含义：达到一定期望收益率情况下，最小化风险

- 数学表达
  $$
  \min_{w} \frac{1}{2}w'Vw \\
  s.t. \; w'e = E(\widetilde r_p) \\
  w'I = 1 \\
  w为风险资产组合中各资产的权重构成的向量\\
  V为风险资产收益率的方差-协方差矩阵\\
  e为风险资产组合中各资产期望收益率构成的向量\\
  I为单位向量
  $$

##### 3.3 资产组合风险分散化-两个风险资产

- 含义：只要两个风险资产**不是完全正相关**的，那么由它们组成的资产组合的风险-收益机会总是**优于**资产组合中各资产单独的风险-收益机会

- 数学表达
  $$
  期望收益：E(r_p) = w_1 E(r_1) + (1-w_1)E(r_2) \\
  标准差：\sigma_p = \sqrt {w_1^2\sigma_1^2 + w_2^2 \sigma_2^2 + 2w_1w_2cov(\widetilde {r_1}, \widetilde {r_2})} \leq w_1 \sigma_1 + w_2 \sigma_2
  $$

- 

##### 3.4 资产组合风险分散化-N个风险资产的资产组合

- 含义：当$N$较大时，协方差项的数目将远远超过方差项，此时，资产组合的风险将主要由资产收益率的协方差大小决定，而资产自身的风险水平可以忽略不计

- 数学表达
  $$
  \sigma_p^2 = \sum_{i=1}^N \sum_{j=1}^N w_i w_j \sigma_{ij} = \sum_{i=1}^N w_i^2 \sigma_i^2 + \sum_{i=1}^N \sum_{j\neq i}^N w_i w_j \sigma_{ij}\\
  假设各资产等比例：\sigma_p^2 = \frac{1}{N} \sigma^2 + (\frac{N^2 - N }{N^2})\rho \sigma^2
  $$

##### 3.5 系统性风险与非系统性风险

| 非系统性风险     | 系统性风险           |
| ---------------- | -------------------- |
| 反映资产本身特性 | 反映各资产的共同运动 |
| 可最终消除       | 无法消除             |
| 个别风险         | 市场风险             |

#### 4. 阅读材料

1.  [基于均值方差模型的柔性资源投资组合策略_杨斌.pdf](基于均值方差模型的柔性资源投资组合策略_杨斌.pdf) 
2.  [具有基数约束的多阶段均值-方差投资组合优化_郝静.pdf](具有基数约束的多阶段均值-方差投资组合优化_郝静.pdf) 
3.  [开环策略下多阶段均值-方差投资组合优化研究_刘德彬.pdf](开环策略下多阶段均值-方差投资组合优化研究_刘德彬.pdf) 

#### 5. 参考资料

1. [美赛第十五次培训-经管类模型概览-均值方差组合模型](https://vshare.sjtu.edu.cn/play/cd8ea54e5f1b42cf7229ef9202c8c9df)
2. [Python实现均值方差组合模型](https://blog.csdn.net/PythonShanyang/article/details/105350629?ops_request_misc=&request_id=&biz_id=102&utm_term=Markowitz%E5%9D%87%E5%80%BC-%E6%96%B9%E5%B7%AE%E6%A8%A1%E5%9E%8B&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-7-105350629.first_rank_v2_pc_rank_v29&spm=1018.2226.3001.4187)

3. [均值方差模型推导与Python代码实现](https://zhuanlan.zhihu.com/p/158994244)