function S = modelset(varargin)
    % Options and defaults
    S = struct();
    S.Optimizer    = 'sgd';   % 'sgd' | 'momentum' | 'rmsprop' | 'adam'
    S.OptimizerFcn = [];

    % Shared
    S.LearningRate = 0.1;

    % Momentum
    S.Momentum     = 0.9;

    % RMSprop
    S.Alpha        = 0.99;
    S.Epsilon      = 1e-8;

    % Adam
    S.Beta1        = 0.9;
    S.Beta2        = 0.999;

    % Parse user overrides
    for i = 1:2:numel(varargin)
        S.(varargin{i}) = varargin{i+1};
    end

    % Build optimizer factory if not provided
    if isempty(S.OptimizerFcn)
        switch lower(string(S.Optimizer))
            case "sgd"
                S.OptimizerFcn = @(params, opts) SGD(params, 'LearningRate', opts.LearningRate);
            case "momentum"
                S.OptimizerFcn = @(params, opts) Momentum(params, 'LearningRate', opts.LearningRate, ...
                    'Momentum',     opts.Momentum);
            case "rmsprop"
                S.OptimizerFcn = @(params, opts) RMSprop(params, 'LearningRate', opts.LearningRate, ...
                    'Alpha',        opts.Alpha, ...
                    'Epsilon',      opts.Epsilon);
            case "adam"
                S.OptimizerFcn = @(params, opts) Adam(params, 'LearningRate', opts.LearningRate, ...
                    'Beta1',        opts.Beta1, ...
                    'Beta2',        opts.Beta2, ...
                    'Epsilon',      opts.Epsilon);
            otherwise
                error('modelset:UnknownOptimizer', 'Unknown optimizer: %s', S.Optimizer);
        end
    end
end