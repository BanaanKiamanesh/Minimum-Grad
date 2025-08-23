classdef (Abstract) Layer < handle
    properties
        Activation
    end

    methods
        function obj = Layer(Activation)
            if nargin < 1
                Activation = [];
            end
            obj.Activation = Activation;
        end
    end

    methods (Abstract)
        Out = Forward(obj, Inp)
        P = Param(obj)
        ZeroGrad(obj)
    end

    methods (Access = protected)
        function Out = ApplyActivation(obj, X)
            if isempty(obj.Activation)
                Out = X;
            else
                if isa(obj.Activation, 'function_handle')
                    Out = obj.Activation(X);
                else
                    s = lower(string(obj.Activation));
                    if s == "relu"
                        Out = Relu(X);
                    elseif s == "sigmoid"
                        Out = Sigmoid(X);
                    elseif s == "tanh"
                        Out = Tanh(X);
                    elseif s == "leakyrelu"
                        Out = LeakyRelu(X);
                    elseif s == "softmax"
                        Out = Softmax(X);
                    else
                        error('Unknown activation');
                    end
                end
            end
        end
    end
end
