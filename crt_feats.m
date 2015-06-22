% ����n_feats������
% ��Ϊfeats�ǱȽ������������Ҫ��ȡ2 * n_feats��������
% ������Ϊ0~1֮�������
% �����Ľ��
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