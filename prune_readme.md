## Personal Agent(Structured Pruning)
### 步骤
1. 聚类数据（LaMP_4），数据格式为 rag + python code format，作为calibration data(nsample <= 128, seqlen = 128)
2. FLAP 计算每一层每个 channel 的 score
3. sort 所有 score，根据目标 sparsity 得到 threshold。其中，由于 Llama3 采用 GQA，一个注意力组含有512 channel，对 channel score 取平均得到一个组的 score
4. score 小于 threshold 的 channel/group 被 pruned
5. post-finetune，数据使用 personal data

### 伪代码（步骤2～4）
- for $l$ in layers: 
    - $S_{:,j}^\ell = \frac{1}{N-1} \sum_{n=1}^N (X_{n,j,:}^\ell - \overline{X}_{:,j,:}^\ell)^2 \cdot ||\mathbf{W}_{:,j}^\ell||_2^2,$ where $j$ is the $j$-th channel in a layer, $X$ is the samples, $W$ is model weights and $S$ is scores
- $S_{attn}$ shape 32*4096, 每512 channel 取平均
- sort([$S_{attn}, S_{mlp}$]), 按 sparsity 得出 threshold （需注意不同 $S$ 中一个值代表的参数量不同）
- Prune channel and attention group
    