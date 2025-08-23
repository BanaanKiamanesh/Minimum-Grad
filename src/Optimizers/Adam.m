classdef Adam < Optimizer
    properties
        LearningRate (1,1) double = 0.001
        Beta1        (1,1) double = 0.9
        Beta2        (1,1) double = 0.999
        Epsilon      (1,1) double = 1e-8
        M                      % cell array, first moment
        V                      % cell array, second raw moment
        T            (1,1) double = 0    % time step
    end

    methods
        function obj = Adam(Params, varargin)
            obj@Optimizer(Params);
            if ~isempty(varargin)
                for i = 1:2:numel(varargin)
                    switch lower(varargin{i})
                        case 'learningrate', obj.LearningRate = varargin{i+1};
                        case 'beta1',        obj.Beta1        = varargin{i+1};
                        case 'beta2',        obj.Beta2        = varargin{i+1};
                        case 'epsilon',      obj.Epsilon      = varargin{i+1};
                    end
                end
            end
            obj.ResetState();
        end

        function ResetState(obj)
            n = numel(obj.Params);
            obj.M = cell(1, n);
            obj.V = cell(1, n);
            for k = 1:n
                p = obj.Params{k};
                obj.M{k} = zeros(size(p.Data));
                obj.V{k} = zeros(size(p.Data));
            end
            obj.T = 0;
        end

        function Step(obj)
            lr = obj.LearningRate; b1 = obj.Beta1; b2 = obj.Beta2; eps = obj.Epsilon;
            obj.T = obj.T + 1;
            b1c = 1 - b1^obj.T;
            b2c = 1 - b2^obj.T;
            for k = 1:numel(obj.Params)
                p = obj.Params{k};
                if ~isempty(p.Grad)
                    g = p.Grad.Data;
                    m = obj.M{k}; v = obj.V{k};
                    m = b1 .* m + (1 - b1) .* g;
                    v = b2 .* v + (1 - b2) .* (g .^ 2);
                    obj.M{k} = m; obj.V{k} = v;

                    mhat = m ./ b1c;
                    vhat = v ./ b2c;
                    p.Data = p.Data - lr .* (mhat ./ (sqrt(vhat) + eps));
                end
            end
        end
    end
end