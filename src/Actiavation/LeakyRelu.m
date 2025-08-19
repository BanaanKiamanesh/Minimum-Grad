function Out = LeakyRelu(T, Leak)
    % LeakyReLU: max(t, 0) + leak * min(t, 0)
    % Leak Default = 0.05

    if nargin < 2 || isempty(Leak)
        Leak = 0.05;
    end
    
    if ~isscalar(Leak) || ~isreal(Leak)
        error('LeakyRelu:InvalidLeak', 'leak must be a real scalar.');
    end

    if ~isa(T, 'Tensor')
        try
            T = Tensor(T);
        catch ME
            error('LeakyRelu:InvalidInput', ...
                'Input must be a Tensor or convertible numeric array. Details: %s', ME.message);
        end
    end

    Data = max(T.Data, 0) + Leak * min(T.Data, 0);

    Req = T.RequireGrad;
    if Req
        Dep = {struct('Tensor', T, 'GradFun', @(g) g .* ((T.Data > 0) + Leak .* (T.Data <= 0)))};
    else
        Dep = {};
    end

    Out = Tensor(Data, Req, Dep);
end
