classdef RMSprop < Optimizer
    properties
        LearningRate (1,1) double = 0.1
        Alpha        (1,1) double = 0.99
        Epsilon      (1,1) double = 1e-8
        SqAvg                 % cell array, EMA of squared grads
    end

    methods
        function obj = RMSprop(Params, varargin)
            obj@Optimizer(Params);
            if ~isempty(varargin)
                for i = 1:2:numel(varargin)
                    switch lower(varargin{i})
                        case 'learningrate', obj.LearningRate = varargin{i+1};
                        case 'alpha',        obj.Alpha        = varargin{i+1};
                        case 'epsilon',      obj.Epsilon      = varargin{i+1};
                    end
                end
            end
            obj.ResetState();
        end

        function ResetState(obj)
            n = numel(obj.Params);
            obj.SqAvg = cell(1, n);
            for k = 1:n
                p = obj.Params{k};
                obj.SqAvg{k} = zeros(size(p.Data));
            end
        end

        function Step(obj)
            lr = obj.LearningRate; a = obj.Alpha; eps = obj.Epsilon;
            for k = 1:numel(obj.Params)
                p = obj.Params{k};
                if ~isempty(p.Grad)
                    g = p.Grad.Data;
                    v = obj.SqAvg{k};
                    v = a .* v + (1 - a) .* (g .^ 2);
                    obj.SqAvg{k} = v;
                    p.Data = p.Data - lr .* (g ./ (sqrt(v) + eps));
                end
            end
        end
    end
end