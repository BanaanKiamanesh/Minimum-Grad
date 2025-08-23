classdef SGD < Optimizer
    properties
        LearningRate (1,1) double = 0.1
    end

    methods
        function obj = SGD(Params, varargin)
            obj@Optimizer(Params);
            if ~isempty(varargin)
                for i = 1:2:numel(varargin)
                    if strcmpi(varargin{i}, 'LearningRate')
                        obj.LearningRate = varargin{i+1};
                    end
                end
            end
        end

        function Step(obj)
            lr = obj.LearningRate;
            for k = 1:numel(obj.Params)
                p = obj.Params{k};
                
                if ~isempty(p.Grad)
                    p.Data = p.Data - lr .* p.Grad.Data;
                end
            end
        end
    end
end
