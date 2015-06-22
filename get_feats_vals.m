% 给定一幅图像和特征点坐标, 计算特征值
% 这里的参数img_size是为了保证img被向量化的时候以img_size为准

function vals = get_feats_vals(img, feats, img_size)
    if (nargin < 3)
        assert(ndims(img) == 2);
        img_size = size(img);
    end
    n_rows = img_size(0);
    n_cols = img_size(1);
    ys = max(1, min(round(feats(:, 1) * n_rows), n_rows));
    xs = max(1, min(round(feats(:, 2) * n_cols), n_cols));
    ys_a = ys(1:2:end);
    ys_b = ys(2:2:end);
    xs_a = xs(1:2:end);
    xs_b = xs(2:2:end);
    ind_a = sub2ind([n_rows, n_cols], ys_a, xs_a);
    ind_b = sub2ind([n_rows, n_cols], ys_b, xs_b);
    vals = img(ind_a) < img(ind_b);
end