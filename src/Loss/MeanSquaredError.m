function Out = MeanSquaredError(Preds, Actual)
    % Mean Squared Error

    if ~isa(Preds, 'Tensor'); Preds = Tensor(Preds); end
    if ~isa(Actual, 'Tensor'); Actual = Tensor(Actual); end

    Err = Preds - Actual;
    Sum = sum(Err .^ 2);
    N   = numel(Err.Data);

    Out = Sum .* Tensor(1.0 / N);
end
