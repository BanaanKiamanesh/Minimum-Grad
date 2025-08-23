classdef Tensor < handle
    % Simple AutoGrad Tensor with Broadcasting-Aware Back propagation.
    % Supports: sum, log, exp, unary -, +, -, .*, *, .^, slicing, backward.

    properties
        Data
        RequireGrad (1, 1) logical = false
        DependsOn                            % Cell Array of Structs('Tensor', Tensor, 'GradFun', @f)
        Grad                                 % Tensor (Same Shape as Data)
        Shape                                % Size Vector
    end

    methods
        function obj = Tensor(Data, RequireGrad, DependsOn, varargin)
            if nargin < 1
                Data = [];
            end

            obj.Data = Tensor.EnsureArray(Data);

            Init = [];
            if ~isempty(varargin)
                for i = 1:2:numel(varargin)
                    if strcmpi(varargin{i}, 'Init')
                        Init = varargin{i+1};
                    end
                end
            end

            if ~isempty(Init)
                sz = size(obj.Data);
                if isempty(obj.Data) || (isvector(Data) && ~isscalar(obj.Data))
                    s = double(Data(:)).';
                    if isempty(s)
                        s = [1 1];
                    end
                    if numel(s) == 1
                        s = [s 1];
                    end
                    sz = s;
                end

                switch lower(Init)
                    case 'xavier'
                        obj.Data = Tensor.XavierInit(sz);
                    case 'he'
                        obj.Data = Tensor.HeInit(sz);
                    case 'zero'
                        obj.Data = Tensor.ZeroInit(sz);
                    case 'one'
                        obj.Data = Tensor.OneInit(sz);
                    otherwise
                        error('Unknown initializer');
                end
            end

            if nargin >= 2 && ~isempty(RequireGrad)
                obj.RequireGrad = logical(RequireGrad);
            end

            if nargin >= 3 && ~isempty(DependsOn)
                obj.DependsOn = DependsOn;
            else
                obj.DependsOn = {};
            end

            obj.Shape = size(obj.Data);
            obj.Grad  = [];
            if obj.RequireGrad
                obj.ZeroGrad();
            end
        end

        function ZeroGrad(obj)
            obj.Grad = Tensor(zeros(size(obj.Data)));
        end

        function Out = sum(T, varargin)
            Data = sum(T.Data, 'all');
            Req  = T.RequireGrad;

            if Req
                Dep = {struct('Tensor', T, 'GradFun', @(g) g .* ones(size(T.Data)))};
            else
                Dep = {};
            end

            Out = Tensor(Data, Req, Dep);
        end

        function Out = log(T)
            Data = log(T.Data);
            Req  = T.RequireGrad;

            if Req
                dep = {struct('Tensor', T, 'GradFun', @(g) g ./ T.Data)};
            else
                dep = {};
            end

            Out = Tensor(Data, Req, dep);
        end

        function Out = exp(T)
            Data = exp(T.Data);
            Req  = T.RequireGrad;

            if Req
                Dep = {struct('Tensor', T, 'GradFun', @(g) g .* Data)};
            else
                Dep = {};
            end

            Out = Tensor(Data, Req, Dep);
        end

        function Out = uminus(T)
            T = Tensor.EnsureTensor(T);
            Data = -T.Data;
            Req  =  T.RequireGrad;

            if Req
                Dep = {struct('Tensor', T, 'GradFun', @(g) -g)};
            else
                Dep = {};
            end

            Out = Tensor(Data, Req, Dep);
        end

        function Out = plus(T1, T2)
            T1 = Tensor.EnsureTensor(T1);
            T2 = Tensor.EnsureTensor(T2);

            Data = T1.Data + T2.Data;
            Req  = T1.RequireGrad || T2.RequireGrad;
            Dep  = {};

            if T1.RequireGrad
                Dep{end + 1} = struct('Tensor', T1, 'GradFun', @(g) Tensor.ReduceGrad(g, size(T1.Data)));
            end
            if T2.RequireGrad
                Dep{end + 1} = struct('Tensor', T2, 'GradFun', @(g) Tensor.ReduceGrad(g, size(T2.Data)));
            end

            Out = Tensor(Data, Req, Dep);
        end

        function Out = minus(T1, T2)
            T1 = Tensor.EnsureTensor(T1);
            T2 = Tensor.EnsureTensor(T2);

            Out = plus(T1, uminus(T2));
        end

        function Out = times(T1, T2)
            T1 = Tensor.EnsureTensor(T1);
            T2 = Tensor.EnsureTensor(T2);

            Data = T1.Data .* T2.Data;
            Req  = T1.RequireGrad || T2.RequireGrad;
            Dep  = {};

            if T1.RequireGrad
                Dep{end + 1} = struct('Tensor', T1, 'GradFun', @(g) Tensor.ReduceGrad(g .* T2.Data, size(T1.Data)));
            end

            if T2.RequireGrad
                Dep{end + 1} = struct('Tensor', T2, 'GradFun', @(g) Tensor.ReduceGrad(g .* T1.Data, size(T2.Data)));
            end

            Out = Tensor(Data, Req, Dep);
        end

        function Out = mtimes(T1, T2)
            T1 = Tensor.EnsureTensor(T1);
            T2 = Tensor.EnsureTensor(T2);

            Data = T1.Data * T2.Data;
            Req = T1.RequireGrad || T2.RequireGrad;
            Dep = {};

            if T1.RequireGrad
                Dep{end + 1} = struct('Tensor', T1, 'GradFun', @(g) g * T2.Data.');
            end

            if T2.RequireGrad
                Dep{end + 1} = struct('Tensor', T2, 'GradFun', @(g) T1.Data.' * g);
            end

            Out = Tensor(Data, Req, Dep);
        end

        function Out = power(T, p)
            T = Tensor.EnsureTensor(T);

            if isa(p, 'Tensor')
                error('Exponent must be numeric for this simple autograd.');
            end

            if isscalar(p) && p == -1
                Data = 1 ./ T.Data;
            else
                Data = T.Data .^ p;
            end

            Req = T.RequireGrad;
            if Req
                if isscalar(p) && p == -1
                    GradF = @(g) g .* (-1) .* (T.Data .^ -2);
                else
                    GradF = @(g) g .* p .* (T.Data .^ (p - 1));
                end
                Dep = {struct('Tensor', T, 'GradFun', GradF)};
            else
                Dep = {};
            end

            Out = Tensor(Data, Req, Dep);
        end

        function varargout = subsref(T, S)
            if strcmp(S(1).type, '()')
                idx  = S(1).subs;
                Data = T.Data(idx{:});
                Req  = T.RequireGrad;

                if Req
                    GradFun = @(g) Tensor.SliceBackProp(g, size(T.Data), idx);
                    Dep = {struct('Tensor', T, 'GradFun', GradFun)};
                else
                    Dep = {};
                end
                Out = Tensor(Data, Req, Dep);

                if numel(S) > 1
                    [varargout{1:nargout}] = builtin('subsref', Out, S(2:end));
                else
                    if nargout > 0
                        varargout{1} = Out;
                    else
                        assignin('caller', 'ans', Out);
                    end
                end
            else
                [varargout{1:nargout}] = builtin('subsref', T, S);
            end
        end

        function Backward(T, Grad)
            if nargin < 2 || isempty(Grad)
                if isscalar(T.Data)
                    Grad = Tensor(1.0);
                else
                    error('grad must be specified for non-scalar tensor');
                end
            else
                Grad = Tensor.EnsureTensor(Grad);
            end

            if isempty(T.Grad)
                T.Grad = Tensor(zeros(size(T.Data)));
            end

            T.Grad.Data = T.Grad.Data + Grad.Data;

            % Recursively Take the Grad Down to the Last Leaves
            for k = 1:numel(T.DependsOn)
                Dep = T.DependsOn{k};
                BackGrad = Dep.GradFun(Grad.Data);
                Dep.Tensor.Backward(BackGrad);
            end
        end
    end

    methods (Static, Access = private)
        function a = EnsureArray(A)
            if isa(A, 'Tensor')
                a = A.Data;
            else
                a = double(A);
            end
        end

        function t = EnsureTensor(T)
            if isa(T, 'Tensor')
                t = T;
            else
                t = Tensor(T);
            end
        end

        function A = XavierInit(Shape)
            s = double(Shape);
            if isempty(s)
                s = [1 1];
            end
            if numel(s) == 1
                s = [s 1];
            end
            A = randn(s) .* sqrt(1 ./ s(1));
        end

        function A = HeInit(Shape)
            s = double(Shape);
            if isempty(s)
                s = [1 1];
            end
            if numel(s) == 1
                s = [s 1];
            end
            A = randn(s) .* sqrt(2 ./ s(1));
        end

        function A = ZeroInit(Shape)
            s = double(Shape);
            if isempty(s)
                s = [1 1];
            end
            if numel(s) == 1
                s = [s 1];
            end
            A = zeros(s);
        end

        function A = OneInit(Shape)
            s = double(Shape);
            if isempty(s)
                s = [1 1];
            end
            if numel(s) == 1
                s = [s 1];
            end
            A = ones(s);
        end

        function Bigger = SliceBackProp(Grad, FullSize, idx)
            Bigger = zeros(FullSize);
            Bigger(idx{:}) = Grad;
        end
    end

    methods (Static, Access = private)
        function g = ReduceGrad(Grad, ShapeOriginal)
            g  = Grad;
            so = double(ShapeOriginal);
            go = size(g);

            La = numel(so);
            Lg = numel(go);

            if Lg < La
                go = [ones(1, La - Lg), go];
                g  = reshape(g, go);
                Lg = La;
            end

            DimAdded = max(Lg - La, 0);
            if DimAdded > 0
                go = size(g);
                LeadProd = prod(go(1:DimAdded));
                Rest     = go(DimAdded+1:end);
                if isempty(Rest), Rest = 1; end
                g = reshape(g, [LeadProd, Rest]);
                g = sum(g, 1);
                g = reshape(g, [ones(1, DimAdded), Rest]);
            end

            go = size(g);
            if numel(go) < DimAdded + La
                go(end+1:DimAdded+La) = 1;
            end

            for i = 1:La
                dim = DimAdded + i;
                if so(i) == 1 && go(dim) ~= 1
                    g  = sum(g, dim);
                    go = size(g);
                    if numel(go) < DimAdded + La
                        go(end+1:DimAdded+La) = 1;
                    end
                end
            end

            g = reshape(g, so);
        end
    end
end
