function Out = Sigmoid(T)
    % Sigmoid: 1 ./ (1 + exp(-t))

    if ~isa(T, 'Tensor')
        try
            T = Tensor(T);
        catch ME
            error('Sigmoid:InvalidInput', ...
                  'Input must be a Tensor or convertible numeric array. Details: %s', ME.message);
        end
    end

    one = Tensor(ones(size(T.Data), 'like', T.Data));
    den = one + exp(-T);                                % 1 + exp(-t)
    Out = one .* (den .^ -1);                           % 1 .* (denom ^ -1)
end
