function Out = Softmax(T)
    % Softmax Along the Last Dimension for a 2D Tensor (N x C)

    if ~isa(T, 'Tensor')
        try
            T = Tensor(T);
        catch ME
            error('Softmax:InvalidInput', ...
                'Input must be a Tensor or convertible numeric array. Details: %s', ME.message);
        end
    end

    sz     = size(T.Data);

    expT   = exp(T);                                         % N x C
    OneCol = Tensor(ones(sz(2), 1, 'like', T.Data));         % C x 1
    sums   = expT * OneCol;                                  % N x 1 (Summing by Timing it by a One Vector)

    Out    = expT .* (sums .^ -1);                           % broadcast to N x C
end
