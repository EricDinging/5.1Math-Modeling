[toc]

#### 模型-评价主题-统计类评价-威尔科克森符号秩检验【gyj】

##### 1. 模型名称

威尔科克森符号秩经检验(Wilcoxon's Sign Rank Test)

##### 2. 适用范围

- 对于一组样本，可以通过威尔科克森符号秩检验**判断<u>给定值</u>与<u>样本中位数</u>的大小关系**。
- 对于两组已经配对[^1]的样本：用威尔科克森符号秩检验**判断样本的分布情况如何，是否服从同一个分布**。

##### 3. 形式

- <u>单一样本</u>或<u>配对样本</u>

- 样本大小：

  - 小样本($\ n\leq 50\ $)：查阅符号秩检验表
  - 大样本($\ n>50\ $)：查阅正态分布表

  【注】也有文献采用25作为区分大小样本的临界值，但无论如何，当样本数超过50就可以直接考虑大样本情况了。

##### 4. 求解方法

4.1 概念

- 单一样本的中位数检验：
  - 对于一组样本$\ X=(x_1,x_2,\cdots)\ $，如果我们想知道其**中位数是否为某个确定的值$\ m_0\ $**或者**两者之间的大小关系**，可以采用威尔科克森符号秩检验。
- 配对样本的分布比较检验：
  - 对于两组样本数相同的变量$\ X=(x_1,x_2,\cdots),Y=(y_1,y_2,\cdots)\ $

4.2 步骤

###### 4.2.1 小样本 - 单一样本的中位数检验

- **提出假设**：设总体的中位数位$\ Me\ $,要检验$\ m_0\ $与$\ Me\ $之间的大小关系。

  - $\ 双尾检验:\begin{cases}
    H_0:Me=m_0\\
    H_1:Me\neq m_0
    \end{cases}\ $
  - $\ 左尾检验:\begin{cases}
    H_0:Me=m_0\\
    H_1:Me< m_0
    \end{cases}\ $
  - $\ 右尾检验:\begin{cases}
    H_0:Me=m_0\\
    H_1:Me> m_0
    \end{cases}\ $

  上面三种假设分别对应了三种大小关系，遇到具体的题目选择合适的假设即可

- **取显著性水平**：一般令$\ \alpha=0.05\ $

- **对样本中元素分配秩**：设$\ D_i=x_i-m_0(i=1,2,\cdots,n)\ $其中n为删除$\ D_i=0\ $后的样本数。将$\ D_i\ $的<u>绝对值</u>从<u>小到大</u>排序，即从1到n的序号(若有$\ D_i\ $相同的其概况，则将对应的秩求和取均值。) 令$\ T^+\ $表示符号为正的$\ D_i\ $的秩之和，$\ T^-\ $表示符号为负的$\ D_i\ $的秩之和。

- **计算检验统计量**：
  $$
  T=\begin{cases}
  min(T^+,T^-)\quad \quad \quad 双尾检验\\
  T^+\quad \quad \quad \quad \quad \quad \quad  左尾检验\\
  T^-\quad \quad \quad \quad \quad \quad \quad  右尾检验
  \end{cases}
  $$

- **确定拒绝域**：查找[WILCOXON符号秩和检验的T临界值表 ](https://wenku.baidu.com/view/4892cafd7dd184254b35eefdc8d376eeafaa1729.html)符号秩检验表得到临界值$\ T_{\alpha/2,n}\ $(双尾检验)，或$\ T_{\alpha,n}\ $(单尾检验)
  $$
  W=\begin{cases}
  \{T|T\leq T_{\alpha/2,n}\} \quad 双尾检验\\
  \{T|T\leq T_{\alpha,n}\} \quad \quad 单尾检验
  \end{cases}
  $$

- **根据求得的拒绝域W决定是否拒绝$\ H_0\ $**

###### 4.2.2 小样本 - 配对样本的分布

- **提出假设**：设总体的中位数位$\ Me\ $,要检验$\ m_0\ $与$\ Me\ $之间的大小关系。

  - $\ 双尾检验:\begin{cases}
    H_0:X总体分布与Y总体分布相同\\
    H_1:X总体分布与Y总体分布不同
    \end{cases}\ $
  - $\ 左尾检验:\begin{cases}
    H_0:X总体分布与Y总体分布相同\\
    H_1:X总体分布图像咋iY总体分布图像的左方
    \end{cases}\ $
  - $\ 右尾检验:\begin{cases}
    H_0:X总体分布与Y总体分布相同\\
    H_1:X总体分布图像在Y总体分布图像的右放
    \end{cases}\ $

- **取显著性水平$\ \alpha\ $**:一般取0.05

- **对样本中元素分配秩**：设$\ D_i=x_i-m_0(i=1,2,\cdots,n)\ $其中n为删除$\ D_i=0\ $后的样本数。将$\ D_i\ $的<u>绝对值</u>从<u>小到大</u>排序，即从1到n的序号(若有$\ D_i\ $相同的其概况，则将对应的秩求和取均值。) 令$\ T^+\ $表示符号为正的$\ D_i\ $的秩之和，$\ T^-\ $表示符号为负的$\ D_i\ $的秩之和。

- **计算检验统计量**：
  $$
  T=\begin{cases}min(T^+,T^-)\quad \quad \quad 双尾检验\\T^+\quad \quad \quad \quad \quad \quad \quad  左尾检验\\T^-\quad \quad \quad \quad \quad \quad \quad  右尾检验\end{cases}
  $$

- **确定拒绝域**：查找符号秩检验表得到临界值$\ T_{\alpha/2,n}\ $(双尾检验)，或$\ T_{\alpha,n}\ $(单尾检验)
  $$
  W=\begin{cases}\{T|T\leq T_{\alpha/2,n}\} \quad 双尾检验\\\{T|T\leq T_{\alpha,n}\} \quad \quad 单尾检验\end{cases}
  $$

- **根据求得的拒绝域W决定是否拒绝$\ H_0\ $**

###### 4.2.3 大样本 - 单一样本的符号秩检验

前面的过程都与小样本相同，在计算检验统计量时开始不同。步骤如下

- 计算检验统计量：
  $$
  T=\begin{cases}min(T^+,T^-)\quad \quad \quad 双尾检验\\T^+\quad \quad \quad \quad \quad \quad \quad  左尾检验\\T^-\quad \quad \quad \quad \quad \quad \quad  右尾检验\end{cases}
  $$
  由于样本容量较大，此时的T近似服从正态分布，且有
  $$
  E(T)=\frac{n(n+1)}{4}
  $$

  $$
  D(T)=\frac{n(n+1)(2n+1)}{24}
  $$

  构造检验统计量Z，使得
  $$
  Z=\frac{T-E(T)}{\sqrt{D(T)}}\sim N(0,1)
  $$

- 确定拒绝域：查[标准正态分布的临界值表](https://zhidao.baidu.com/question/1836435409517148420.html)得到临界值$\ Z_{\alpha/2}\ $(双尾检验)或$\ Z_{\alpha}\ $(单尾检验)，确定拒绝域W
  $$
  W=\begin{cases}
  \{Z|Z\geq Z_{\alpha/2}\quad 或\quad Z\leq-Z_{\alpha/2}\}\quad \quad 双尾检验\\
  \{Z|Z\leq-Z_{\alpha}\}\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad左尾检验\\
  \{Z|Z\geq Z_{\alpha}\}\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\quad右尾检验
  \end{cases}
  $$

- **根据求得的拒绝域W决定是否拒绝$\ H_0\ $**

###### 4.2.4 大样本 - 配对样本的分布

前四步与小样本的配对样本分布相同，后三步与4.2.3的步骤相同。

4.3 例子

题目：某位品检员想知道产品数据的中位数是否与之前记录的82克相同，他随机抽取了生产线上14件产品检验，得到如下数据：83.5，81.3，78.2，82，85.4，88.3，76.2，79.8，83.6，77.3，86.8，80.5，81.1，79.6.试在显著性水平$\ \alpha=0.05\ $下用符号检验法进行检验。

- 提出假设：
  $$
  \begin{cases}
  H_0:Me=82\\
  H_1:Me\neq82
  \end{cases}
  $$

- $\ \alpha=0.05\ $

- 对样本元素分配秩：

  |    产品     |  1   |  2   |  3   |  4   |  5   |  6   |  7   |  8   |  9   |  10  |  11  |  12  |  13  |  14  |
  | :---------: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
  |    质量     | 83.5 | 81.3 | 78.2 |  82  | 85.4 | 88.3 | 76.2 | 79.8 | 83.6 | 77.3 | 86.8 | 80.5 | 81.1 | 79.6 |
  |  $\ D_i\ $  | 1.5  | -0.7 | -3.8 |  0   | 3.4  | 6.3  | -5.8 | -2.2 | 1.6  | -4.7 | 4.8  | -1.5 | -0.9 | -2.4 |
  | $\ |D_i|\ $ | 1.5  | 0.7  | 3.8  |  0   | 3.4  | 6.3  | 5.8  | 2.2  | 1.6  | 4.7  | 4.8  | 1.5  | 0.9  | 2.4  |
  |     秩      | 3.5  |  1   |  9   |  -   |  8   |  13  |  12  |  6   |  5   |  10  |  11  | 3.5  |  2   |  7   |

  $$
  T^+=3.5+8+13+5+11=40.5
  $$

  $$
  T^-=1+9+12+6+10+3.5+2+7=50.5
  $$

  注意这里的$\ T^+和T^-\ $是秩的和相加，不是正数(或负数)本身的大小相加

- 计算检验统计量：本题考察“产品数据的中位数与之前记录的82克<u>是否相同</u>”，故采取双尾检验
  $$
  T=min(T^+,T^-)=40.5,n=13
  $$
  

- 确定拒绝域：

  查表得到$\ T_{\alpha/2,n}=T_{0.025,13}=17<T\ $

  可见T不在拒绝域里，故接受$\ H_0\ $，即中位数就是82克。

4.4 代码实现

```matlab
% 产品的质量
quality=[83.5, 81.3, 78.2, 82, 85.4, 88.3, 76.2, 79.8, 83.6, 77.3, 86.8, 80.5, 81.1, 79.6];
% 将产品质量、待检验值、显著性水平、检验模式（单尾/双尾）、样本规模输入wilcoxon()函数，得到检验结果
[ H, n, T, Z, bound, T_p, T_n ] = wilcoxon( quality, 82, 0.05, 'both', 'auto' )

% function [ H, n, T, Z, bound, T_p, T_n ] = wilcoxon( X, morY, alpha, tailType, sizeType )
% Wilcoxon符号秩检验的MATLAB程序代码

% H 表示最终所接受的假设。若为0，表示接受原假设H0；若为1，拒绝原假设H0，接受H1。
% n 样本差值中删除0后剩余的样本数。
% T 检验统计量T。
% Z 检验统计量Z，小样本检验中返回的Z=T。
% bound 拒绝域的边界值。小样本双尾检验时该值为T_{a/2,n}，单尾检验为T_{a,n}。
%                   大样本双尾检验中该值为Z_{a/2}，单尾检验为Z_a。
% T_p,T_n 分别为文档中的T^+,T^-。
% X 第一组样本。
% morY 确定值m0或第二组样本。单一样本中位数检验时为前者，配对样本分布比较检验为后者。
% alpha 显著性水平，默认0.05。
% tailType 验证类型，可取值：
%           'both'：双尾检验（默认）。
%           'left'：左尾检验。
%           'right'：右尾检验。
% sizeType 样本规模类型，可取值：
%           'auto'：自动识别（默认）。
%           'small'：小样本检验。
%           'large'：大样本检验。

% 注意事项：
% 1.X应为一维向量，morY应为标量或一维向量，分别对应单一样本和配对样本两种情况。若为后者，两者规模应相同。
% 2.sizeType默认的自动识别以50为界限，即初始样本规模大于50采用'large'，否则采用'small'。

function [ H, n, T, Z, bound, T_p, T_n ] = wilcoxon( X, morY, alpha, tailType, sizeType )
% function [ H, n, T, Z, bound, T_p, T_n ] = wilcoxon( X, morY, alpha, tailType, sizeType )
% Wilcoxon符号秩检验的MATLAB程序代码

% H 表示最终所接受的假设。若为0，表示接受原假设H0；若为1，拒绝原假设H0，接受H1。
% n 样本差值中删除0后剩余的样本数。
% T 检验统计量T。
% Z 检验统计量Z，小样本检验中返回的Z=T。
% bound 拒绝域的边界值。小样本双尾检验时该值为T_{a/2,n}，单尾检验为T_{a,n}。
%                   大样本双尾检验中该值为Z_{a/2}，单尾检验为Z_a。
% T_p,T_n 分别为文档中的T^+,T^-。
% X 第一组样本。
% morY 确定值m0或第二组样本。单一样本中位数检验时为前者，配对样本分布比较检验为后者。
% alpha 显著性水平，默认0.05。
% tailType 验证类型，可取值：
%           'both'：双尾检验（默认）。
%           'left'：左尾检验。
%           'right'：右尾检验。
% sizeType 样本规模类型，可取值：
%           'auto'：自动识别（默认）。
%           'small'：小样本检验。
%           'large'：大样本检验。

% 注意事项：
% 1.X应为一维向量，morY应为标量或一维向量，分别对应单一样本和配对样本两种情况。若为后者，两者规模应相同。
% 2.sizeType默认的自动识别以50为界限，即初始样本规模大于50采用'large'，否则采用'small'。

%% 以下为一般过程
%% 参数初始化。
if nargin<3
    alpha = 0.05;   %默认显著性水平。
end
if nargin<4
    tailType = 'both';  %默认验证类型。
end
if nargin<5
    sizeType = 'auto';  %默认样本规模类型。
end
if ~isscalar(morY) && numel(X)~=numel(morY) %检验morY合法性。
    error('Bad parameters!');
end
if strcmpi(sizeType, 'auto')    %自动识别样本规模类型。
    if numel(X)<=50
        sizeType = 'small';
    else
        sizeType = 'large';
    end
end

%% 计算样本差值并进行秩的分配。
diffxy = X(:)-morY(:);  %样本差值。
epsdiff = eps(X(:))+eps(morY(:));   %epsilon，用于判断0值。

% 删除0值。
zeromask = abs(diffxy)<=epsdiff;
diffxy(zeromask) = [];

n=length(diffxy);   %删除0后剩余的样本数。

posMask = diffxy>0; %记录差值为正的样本下标。
tieRank = tiedrank(abs(diffxy)); %进行秩的分配，相同则分配均值。

%% 计算检验统计量T以及对概率进行转换。
T_p = sum(tieRank(posMask));    %计算T^+。
T_n = sum(tieRank(~posMask));   %计算T^-。

switch lower(tailType)
    case 'both' %双尾检验。
        T = min(T_p, T_n);  %计算统计量T。
        P = 1-alpha/2;  %用于后续调用的概率。
    case {'left','right'}
        if lower(tailType(1))=='l'  %左尾检验。
            T = T_p;
        else    %右尾检验。
            T = T_n;
        end
        P = 1-alpha;
    otherwise
        error('Unknown tail type!');
end

%% 计算统计量Z，拒绝域边界bound以及给出决定H。
switch lower(sizeType)
    case 'small'    %小样本。
        Z = T;
        bound = wilTable(n, 1-P);   %拒绝域边界，调用函数wilTable。
        H = T<=bound;   %最终决定H。
    case 'large'    %大样本。
        ET = n*(n+1)/4; %计算E(T)。
        DT = n*(n+1)*(2*n+1)/24;    %计算D(T)。
        Z = (T-ET)/sqrt(DT);    %计算统计量Z。
        bound = norminv(P); %拒绝域边界，由正态分布的CDF反函数求出。
        switch lower(tailType)  %最终决定H，三种情况。
            case 'both'
                H = Z>=bound || Z<=-bound;
            case 'left'
                H = Z<=-bound;
            case 'right'
                H = Z>=bound;
        end
    otherwise
        error('Unknown size type!');
end

%% 调用方式
% quality=[83.5, 81.3, 78.2, 82, 85.4, 88.3, 76.2, 79.8, 83.6, 77.3, 86.8, 80.5, 81.1, 79.6];
% [ H, n, T, Z, bound, T_p, T_n ] = wilcoxon( quality, 82, 0.05, 'both', 'auto' )

end


function pbound = wilTable(n, P)
% function pbound = wilTable(n, P)
% 用于计算wilcoxon符号秩检验表，并查出相应值。

maxw = n*(n+1)/2;   %秩和的最大可能值。

%% 由动态规划求出各个秩和出现的概率。
C = zeros(maxw+1,1);    %动态规划数组，储存当前状态下各个秩和组合的数目。
C(1) = 1;   %为0的秩和只有当所有秩均不参与求和一种组合（即样本差值为负值）。
for k = 1:n   %动态规划过程，每个循环内C(J)代表考虑1~k的秩所能组合出秩和J-1的种类数。
    updateMask = (k:maxw)+1;
    C(updateMask) = C(updateMask)+C(updateMask-k);
end
C = C/(2^n);    %将组合数转换为概率。

%% 计算累积概率，同时搜索与目标概率最相近的值。
if C(1)>P
    pbound = 0;
    return;
end
for k = 2:maxw+1
    C(k) = C(k)+C(k-1);   %计算累积概率。
    if C(k)>P   %当大于目标概率时，选择较近的作为最终边界值。
        if abs(C(k)-P)<=abs(C(k-1)-P)
            pbound = k-1;
        else
            pbound = k-2;
        end
        break;
    end
end

end
```



##### 5. 补充资料

1. [数模官网 - 威尔科克森符号秩检验](https://anl.sjtu.edu.cn/mcm/docs/1模型/8评价主题/2统计类评价/威尔科克森符号秩检验/doc)
2. [WILCOXON符号秩和检验的T临界值表](https://wenku.baidu.com/view/4892cafd7dd184254b35eefdc8d376eeafaa1729.html)
3. [标准正态分布的临界值表](https://zhidao.baidu.com/question/1836435409517148420.html)

4. 上文中用到的其他知识点：

- [^1]: 已配对样本：两组变量的样本数要相同，样本间的元素也要两两配对。例，两个评分员对同一个产品的打分就可以配对等。

  