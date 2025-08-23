classdef (Abstract) Optimizer < handle
    properties
        Params
    end

    methods
        function obj = Optimizer(Params)
            if nargin < 1
                Params = {};
            end
            obj.Params = Params;
        end

        % Rebind parameters (e.g., after rebuilding a model) and
        % allow subclasses to reinitialize their state.
        function Rebind(obj, Params)
            obj.Params = Params;
            obj.ResetState();
        end

        % Default no‑op. Subclasses override to (re)allocate state buffers.
        function ResetState(obj)
        end
    end

    methods (Abstract)
        Step(obj)
    end
end