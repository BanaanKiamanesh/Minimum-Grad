function S = modelset(varargin)
    S = struct();
    S.OptimizerFcn = [];
    S.LearningRate = 0.1;

    for i = 1:2:numel(varargin)
        S.(varargin{i}) = varargin{i+1};
    end

    if isempty(S.OptimizerFcn) % Default
        S.OptimizerFcn = @(params, opts) SGD(params, 'LearningRate', opts.LearningRate);
    end
end
