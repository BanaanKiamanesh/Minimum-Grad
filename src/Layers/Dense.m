classdef Dense < Layer
    properties
        Shape
        NeedBias (1, 1) logical = true
        Weight
        Bias
    end

    methods
        function obj = Dense(InChannels, OutChannels, NeedBias, Mode, Activation)
            if nargin < 3 || isempty(NeedBias), NeedBias = true;     end
            if nargin < 4 || isempty(Mode),     Mode     = 'xavier'; end
            if nargin < 5, Activation = []; end

            obj@Layer(Activation);

            obj.Shape = [InChannels, OutChannels];
            obj.NeedBias = NeedBias;

            obj.Weight = Tensor(obj.Shape, true, [], 'Init', Mode);

            if obj.NeedBias
                obj.Bias = Tensor([1, OutChannels], true, [], 'Init', 'zero');
            else
                obj.Bias = [];
            end
        end

        function Out = Forward(obj, Inp)
            Out = Inp * obj.Weight;
            if obj.NeedBias
                Out = Out + obj.Bias;
            end
            Out = obj.ApplyActivation(Out);
        end

        function P = Param(obj)
            if obj.NeedBias
                P = {obj.Weight, obj.Bias};
            else
                P = {obj.Weight};
            end
        end

        function ZeroGrad(obj)
            obj.Weight.ZeroGrad();
            if obj.NeedBias
                obj.Bias.ZeroGrad();
            end
        end
    end
end
