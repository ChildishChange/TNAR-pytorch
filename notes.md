1. tf.stop_gradient(a) 会把 包括 a 和 a 之前所有的节点的连接截断，不计算从这个方向来的梯度
2. 训练时的 loss 包括四部分 ：ce_loss / en_loss / vat_loss / vat_loss_orth 其中 只有 vat_loss 和 vat_loss_orth 与共轭梯度法有关，在代码中被 stop_gradient 了
3. vat_loss 是原始输出和加了噪声的输出的js散度
4. r_adv 是噪声，r_adv_orth 是垂直 r_adv 的噪声
5. 共轭梯度法，目的是求二元函数极值点，这里要求的是 $\frac{1}{2}\eta^T J^T HJ\eta$ 
6. tf.gradient = $\frac{\partial{dist}}{\partial{classifier}}\cdot \frac{\partial{classifier}}{\partial{(x_r-x_{recon})}} \cdot \frac{\partial{(x_r-x_{recon})}}{\partial{r_{adv}}}$
7. 几个比较关键的参数/方法:
    1. a.detach()，使用了a.detach()进行计算的，将会把 a.detach()视为一个常量，不计算 包括 a 与之前所有路径的梯度
    2. torch.autograd() 可以计算任意节点之间的梯度，计算完成后会销毁计算图，使用 retain_graph=True 即可多次求导，被求导节点的 grad 信息不会改变，梯度不会叠加；使用 节点.backward(retain_graph=True) 求导则会叠加梯度
    3. 当需要计算多阶导时，使用 create_graph=True 参数，将新的节点加入计算图后再次计算，这一操作同时会保存之前的计算图
8. 运行时出现了显存持续增长的现象，原因是 normalization 函数，传入参数不能是 requires_grad = True 的张量