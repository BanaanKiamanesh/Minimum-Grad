classdef Model < Layer
    properties
        Layers = {}
        IsTraining (1,1) logical = true
    end

    methods
        function obj = Model(Layers)
            if nargin < 1 || isempty(Layers)
                Layers = {};
            end
            obj.Layers = Layers;
        end

        function obj = Add(obj, L)
            obj.Layers{end+1} = L;
        end

        function Out = Forward(obj, Inp)
            X = Inp;
            for i = 1:numel(obj.Layers)
                X = obj.Layers{i}.Forward(X);
            end
            Out = X;
        end

        function P = Param(obj)
            P = {};
            for i = 1:numel(obj.Layers)
                Pi = obj.Layers{i}.Param();
                P = [P, Pi];
            end
        end

        function ZeroGrad(obj)
            for i = 1:numel(obj.Layers)
                obj.Layers{i}.ZeroGrad();
            end
        end

        function Train(obj)
            obj.IsTraining = true;
        end

        function Eval(obj)
            obj.IsTraining = false;
        end
    end
end
