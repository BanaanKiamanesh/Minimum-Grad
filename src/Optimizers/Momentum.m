classdef Momentum < Optimizer
    properties
        LearningRate (1,1) double = 0.1
        Moment     (1,1) double = 0.9
        Vel                  % cell array, per-parameter velocity
    end

    methods
        function obj = Momentum(Params, varargin)
            obj@Optimizer(Params);
            if ~isempty(varargin)
                for i = 1:2:numel(varargin)
                    switch lower(varargin{i})
                        case 'learningrate', obj.LearningRate = varargin{i+1};
                        case 'momentum',     obj.Moment       = varargin{i+1};
                    end
                end
            end
            obj.ResetState();
        end

        function ResetState(obj)
            n = numel(obj.Params);
            obj.Vel = cell(1, n);
            for k = 1:n
                p = obj.Params{k};
                obj.Vel{k} = zeros(size(p.Data));
            end
        end

        function Step(obj)
            lr = obj.LearningRate; mu = obj.Moment;
            for k = 1:numel(obj.Params)
                p = obj.Params{k};
                if ~isempty(p.Grad)
                    v = obj.Vel{k};
                    v = mu .* v - lr .* p.Grad.Data;
                    obj.Vel{k} = v;
                    p.Data = p.Data + v;
                end
            end
        end
    end
end