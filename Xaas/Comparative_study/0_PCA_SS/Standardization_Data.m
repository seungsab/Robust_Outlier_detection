function X0std=Standardization_Data(X0,mu0,sds0,numobs,SD_type)
switch SD_type
    case 'Mean_centered'
        X0std = (X0 - repmat(mu0,numobs,1));
    case 'Z-score'
        X0std = (X0 - repmat(mu0,numobs,1)) ./ repmat(sds0,numobs,1);
end

end