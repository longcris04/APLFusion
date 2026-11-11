function [fmi_af, fSub] = fmi_block_a(aSub, fSub, w)
% Per-window "a" block of metrics/fmi.m, copied verbatim (lines 118-302 of the
% original, CRLF removed).  Returns fmi_af and fSub as the block leaves it.
        l = round((2*w+1).^2);
        
        if aSub == fSub
            fmi_af = 1;
        else
            aMax = max(aSub(:));
            aMin = min(aSub(:));
            if aMax == aMin
                aSub = ones(2*w+1);
            else
                aSub = (aSub - aMin)/(aMax - aMin);
            end
            
            fMax = max(fSub(:));
            fMin = min(fSub(:));
            if fMax == fMin
                fSub = ones(2*w+1);
            else
                fSub = (fSub - fMin)/(fMax - fMin);
            end
            
            % Normalize the feature images to get the marginal PDFs
            aPdf = aSub(:)./(sum(sum(aSub)));
            fPdf = fSub(:)./(sum(sum(fSub)));
            
            % PDF to CDF tranformation
            aCdf = zeros(size(aPdf));
            fCdf = zeros(size(fPdf));
            aCdf(1) = aPdf(1);
            fCdf(1) = fPdf(1);
            for i = 2:l
                aCdf(i) = aPdf(i)+aCdf(i-1);
                fCdf(i) = fPdf(i)+fCdf(i-1);
            end
            
            % Pearson correlation between marginal PDFs
            aTemp = aPdf - mean(aPdf);
            fTemp = fPdf - mean(fPdf);
            if sum(aTemp.*fTemp) == 0
                c = 0;
            else
                c = sum(aTemp.*fTemp)/sqrt(sum(aTemp.*aTemp)*sum(fTemp.*fTemp));
            end
            
            % Population standard deviations
            e_aPdf = 0; e2_aPdf = 0; e_fPdf = 0; e2_fPdf = 0; 
            for i = 1:l
                e_aPdf  = e_aPdf  + i.*aPdf(i);         % Expected value of aPdf
                e2_aPdf = e2_aPdf + (i.^2).*aPdf(i);    % 2nd-order moment of aPdf
                e_fPdf  = e_fPdf  + i.*fPdf(i);         % Expected value of fPdf
                e2_fPdf = e2_fPdf + (i.^2).*fPdf(i);	% 2nd-order moment of fPdf
            end
            aSd = sqrt(e2_aPdf - e_aPdf.^2);	% Population standard deviation of aPdf
            fSd = sqrt(e2_fPdf - e_fPdf.^2);	% Population standard deviation of fPdf
            
            
            jointEntropy = 0;
            % Joint PDF calculation using simplified Nelsen method with joint entropy calculation
            if c >= 0
                
                if c == 0 || aSd == 0 || fSd == 0
                    phi = 0;   
                else        
                    covUp = 0;
                    for i = 1:l
                        for j = 1:l
                            % Frechet's upper bound bivariate distributions between f & a
                            covUp = covUp + 0.5*(fCdf(i)+aCdf(j)-abs(fCdf(i)-aCdf(j))) - fCdf(i).*aCdf(j);
                        end
                    end
                    corrUp = covUp/(fSd*aSd); % Upper correlation between f & a
                    phi = c/corrUp;
                end
                
                jpdfUp = 0.5*(fCdf(1)+aCdf(1)-abs(fCdf(1)-aCdf(1)));
                jpdf = phi.*jpdfUp + (1-phi).*fPdf(1).*aPdf(1);
                if jpdf~=0
                    jointEntropy = real(-jpdf.*log2(jpdf)); % Joint entropy
                end
                
                % 1-D boundaries
                for i=2:l
                    jpdfUp = 0.5*(fCdf(i)+aCdf(1)-abs(fCdf(i)-aCdf(1))) - ...
                        0.5*(fCdf(i-1)+aCdf(1)-abs(fCdf(i-1)-aCdf(1)));
                    jpdf = phi.*jpdfUp + (1-phi).*fPdf(i).*aPdf(1);
                    if jpdf~=0
                        jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                    end
                end
                for j=2:l
                    jpdfUp = 0.5*(fCdf(1)+aCdf(j)-abs(fCdf(1)-aCdf(j))) - ...
                        0.5*(fCdf(1)+aCdf(j-1)-abs(fCdf(1)-aCdf(j-1)));
                    jpdf = phi.*jpdfUp + (1-phi).*fPdf(1).*aPdf(j);
                    if jpdf~=0
                        jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                    end
                end
                % 2-D walls
                for i=2:l
                    for j=2:l
                        jpdfUp = 0.5*(fCdf(i)+aCdf(j)-abs(fCdf(i)-aCdf(j))) - ...
                            0.5*(fCdf(i-1)+aCdf(j)-abs(fCdf(i-1)-aCdf(j))) - ...
                            0.5*(fCdf(i)+aCdf(j-1)-abs(fCdf(i)-aCdf(j-1))) + ...
                            0.5*(fCdf(i-1)+aCdf(j-1)-abs(fCdf(i-1)-aCdf(j-1)));
                        jpdf = phi.*jpdfUp + (1-phi).*fPdf(i).*aPdf(j);
                        if jpdf~=0
                            jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                        end
                    end
                end
                
            end

            if c < 0
                
                if aSd == 0 || fSd == 0
                    theta = 0;
                else
                    covLo = 0;
                    for i = 1:l
                        for j = 1:l
                            % Frechet's lower bound bivariate distributions between f & a
                            covLo = covLo + 0.5*(fCdf(i)+aCdf(j)-1+abs(fCdf(i)+aCdf(j)-1)) - fCdf(i).*aCdf(j);
                        end
                    end
                    corrLo = covLo/(fSd*aSd); % Lower correlation between f & a
                    theta = c/corrLo;
                end
                
                jpdfLo = 0.5*(fCdf(1)+aCdf(1)-1+abs(fCdf(1)+aCdf(1)-1));
                jpdf = theta.*jpdfLo + (1-theta).*fPdf(1).*aPdf(1);
                if jpdf~=0
                    jointEntropy = real(-jpdf.*log2(jpdf)); % Joint entropy
                end
                
                % 1-D boundaries
                for i=2:l
                    jpdfLo = 0.5*(fCdf(i)+aCdf(1)-1+abs(fCdf(i)+aCdf(1)-1)) - ...
                        0.5*(fCdf(i-1)+aCdf(1)-1+abs(fCdf(i-1)+aCdf(1)-1));
                    jpdf = theta.*jpdfLo + (1-theta).*fPdf(i).*aPdf(1);
                    if jpdf~=0
                        jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                    end
                end
                for j=2:l
                    jpdfLo = 0.5*(fCdf(1)+aCdf(j)-1+abs(fCdf(1)+aCdf(j)-1)) - ...
                        0.5*(fCdf(1)+aCdf(j-1)-1+abs(fCdf(1)+aCdf(j-1)-1));
                    jpdf = theta.*jpdfLo + (1-theta).*fPdf(1).*aPdf(j);
                    if jpdf~=0
                        jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                    end
                end
                % 2-D walls
                for i=2:l
                    for j=2:l
                        jpdfLo = 0.5*(fCdf(i)+aCdf(j)-1+abs(fCdf(i)+aCdf(j)-1)) - ...
                            0.5*(fCdf(i-1)+aCdf(j)-1+abs(fCdf(i-1)+aCdf(j)-1)) - ...
                            0.5*(fCdf(i)+aCdf(j-1)-1+abs(fCdf(i)+aCdf(j-1)-1)) + ...
                            0.5*(fCdf(i-1)+aCdf(j-1)-1+abs(fCdf(i-1)+aCdf(j-1)-1));
                        jpdf = theta.*jpdfLo + (1-theta).*fPdf(i).*aPdf(j);
                        if jpdf~=0
                            jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                        end
                    end
                end
                
            end
            
            % Marginal entropies
            index = find(aPdf~=0);
            aEntropy = sum(-aPdf(index).*log2(aPdf(index)));
            index = find(fPdf~=0);
            fEntropy = sum(-fPdf(index).*log2(fPdf(index)));
            
            % Mutual information between a & f
            mi = aEntropy + fEntropy - jointEntropy;  
            
            % Overall normalized mutual information
            if mi == 0
                fmi_af = 0;
            else
                fmi_af = 2.*mi./(aEntropy+fEntropy);
            end
            
        end
end
