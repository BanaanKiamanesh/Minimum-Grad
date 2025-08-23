classdef (Abstract) Optimizer < handle
    properties
        Params
    end

    methods
        function obj = Optimizer(Params)
            obj.Params = Params;
        end
    end

    methods (Abstract)
        Step(obj)
    end
end
