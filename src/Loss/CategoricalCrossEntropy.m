function Out = CategoricalCrossEntropy(Preds, Actual)
    % Categorical Cross-Entropy (with Logits)

    if ~isa(Preds, 'Tensor');  Preds = Tensor(Preds);   end
    if ~isa(Actual, 'Tensor'); Actual = Tensor(Actual); end

    P   = Softmax(Preds);
    Sum = sum(Actual .* log(P));
    N   = size(Actual.Data, 1);

    Out = (-Sum) .* Tensor(1.0 / N);
end
