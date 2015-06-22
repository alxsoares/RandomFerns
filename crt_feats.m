% 创建n_feats个特征
% 因为feats是比较特征，因此需要提取2 * n_feats个特征点
% 特征点为0~1之间的数字
% 创建的结果
%     y_1a, x_1a
%     y_1b, x_1b
%     y_2a, x_2a
%     y_2b, x_2b
%     ...

function feats = crt_feats(n_feats)
    n_samples = 2 * n_feats;
    ys = rand(n_samples, 1);
    xs = rand(n_samples, 1);
    feats = [ys, xs];
end