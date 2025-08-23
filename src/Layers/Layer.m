classdef (Abstract) Layer < handle
    methods (Abstract)
        Out = Forward(obj, Inp)
        P = Param(obj)
        ZeroGrad(obj)
    end
end
