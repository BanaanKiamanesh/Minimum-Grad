function Out = Tanh(T)
    % Tanh: (exp(t) - exp(-t)) ./ (exp(t) + exp(-t))

    if ~isa(T, 'Tensor')
        try
            T = Tensor(T);
        catch ME
            error('Tanh:InvalidInput', ...
                'Input must be a Tensor or convertible numeric array. Details: %s', ME.message);
        end
    end

    expT   = exp(T);
    expTn  = exp(-T);

    num  = expT - expTn;
    den  = expT + expTn;

    Out  = num .* (den .^ -1);
end
