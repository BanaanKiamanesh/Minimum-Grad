function Out = Relu(T)
    % ReLU: max(t, 0)

    if ~isa(T, 'Tensor')
        try
            T = Tensor(T);
        catch ME
            error('Relu:InvalidInput', ...
                'Input must be a Tensor or convertible numeric array. Details: %s', ME.message);
        end
    end

    Data = max(T.Data, 0);

    Req = T.RequireGrad;
    if Req
        Dep = {struct('Tensor', T, 'GradFun', @(g) g .* (T.Data > 0))};
    else
        Dep = {};
    end

    Out = Tensor(Data, Req, Dep);
end
