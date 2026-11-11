function fmi_bf = fmi_block_b(bSub, fSub, w)
% Per-window "b" block of metrics/fmi.m, copied verbatim (lines 304-489 of the
% original, CRLF removed).
        if bSub == fSub
            fmi_bf = 1;
        else
            bMax = max(bSub(:));
            bMin = min(bSub(:));
            if bMax == bMin
                bSub = ones(2*w+1);
            else
                bSub = (bSub - bMin)/(bMax - bMin);
            end

            fMax = max(fSub(:));
            fMin = min(fSub(:));
            if fMax == fMin
                fSub = ones(2*w+1);
            else
                fSub = (fSub - fMin)/(fMax - fMin);
            end

            % PDF of the gradients of the images
            bPdf = bSub(:)./(sum(sum(bSub)));
            fPdf = fSub(:)./(sum(sum(fSub)));

            l = length(bPdf);

            % PDF to CDF tranformation
            bCdf = zeros(size(bPdf));
            fCdf = zeros(size(fPdf));
            bCdf(1) = bPdf(1);
            fCdf(1) = fPdf(1);
            for i=2:l
                bCdf(i) = bPdf(i)+bCdf(i-1);
                fCdf(i) = fPdf(i)+fCdf(i-1);
            end

            % Pearson correlation between marginal PDFs
            bTemp = bPdf - mean(bPdf);
            fTemp = fPdf - mean(fPdf);
            if sum(bTemp.*fTemp) == 0
                c = 0;
            else
                c = sum(bTemp.*fTemp)/sqrt(sum(bTemp.*bTemp)*sum(fTemp.*fTemp));
            end

            % Population standard deviations
            e_bPdf = 0; e2_bPdf = 0; e_fPdf = 0; e2_fPdf = 0; 
            for i = 1:l
                e_bPdf  = e_bPdf  + i.*bPdf(i);         % Expected value of the aPdf
                e2_bPdf = e2_bPdf + (i.^2).*bPdf(i);	% 2nd-order moment of the aPdf
                e_fPdf  = e_fPdf  + i.*fPdf(i);         % Expected value of the bPdf
                e2_fPdf = e2_fPdf + (i.^2).*fPdf(i);	% 2nd-order moment of the bPdf
            end
            bSd = sqrt(e2_bPdf - e_bPdf.^2);	% Population standard deviation of the intensity
            fSd = sqrt(e2_fPdf - e_fPdf.^2);	% Population standard deviation of the intensity


            jointEntropy = 0;
            % JPDF calculation using simplified Nelsen method with joint entropy calculation
            if c >= 0

                if c == 0 || bSd == 0 || fSd == 0
                    phi = 0;   
                else        
                    covUp = 0;
                    for i = 1:l
                        for j = 1:l
                            % Frechet's upper bound bivariate distributions between f & b
                            covUp = covUp + 0.5*(fCdf(i)+bCdf(j)-abs(fCdf(i)-bCdf(j))) - fCdf(i).*bCdf(j);
                        end
                    end
                    corrUp = covUp/(fSd*bSd); % Upper correlation between f & b
                    phi = c/corrUp;
                end

                jpdfUp = 0.5*(fCdf(1)+bCdf(1)-abs(fCdf(1)-bCdf(1)));
                jpdf = phi.*jpdfUp + (1-phi).*fPdf(1).*bPdf(1);
                if jpdf~=0
                    jointEntropy = real(-jpdf.*log2(jpdf)); % Joint entropy
                end

                % 1-D boundaries
                for i=2:l
                    jpdfUp = 0.5*(fCdf(i)+bCdf(1)-abs(fCdf(i)-bCdf(1))) - ...
                        0.5*(fCdf(i-1)+bCdf(1)-abs(fCdf(i-1)-bCdf(1)));
                    jpdf = phi.*jpdfUp + (1-phi).*fPdf(i).*bPdf(1);
                    if jpdf~=0
                        jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                    end
                end
                for j=2:l
                    jpdfUp = 0.5*(fCdf(1)+bCdf(j)-abs(fCdf(1)-bCdf(j))) - ...
                        0.5*(fCdf(1)+bCdf(j-1)-abs(fCdf(1)-bCdf(j-1)));
                    jpdf = phi.*jpdfUp + (1-phi).*fPdf(1).*bPdf(j);
                    if jpdf~=0
                        jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                    end
                end
                % 2-D walls
                for i=2:l
                    for j=2:l
                        jpdfUp = 0.5*(fCdf(i)+bCdf(j)-abs(fCdf(i)-bCdf(j))) - ...
                            0.5*(fCdf(i-1)+bCdf(j)-abs(fCdf(i-1)-bCdf(j))) - ...
                            0.5*(fCdf(i)+bCdf(j-1)-abs(fCdf(i)-bCdf(j-1))) + ...
                            0.5*(fCdf(i-1)+bCdf(j-1)-abs(fCdf(i-1)-bCdf(j-1)));
                        jpdf = phi.*jpdfUp + (1-phi).*fPdf(i).*bPdf(j);
                        if jpdf~=0
                            jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                        end
                    end
                end

            end

            if c < 0

                if bSd == 0 || fSd == 0
                    theta = 0;
                else
                    covLo = 0;
                    for i = 1:l
                        for j = 1:l
                            % Frechet's lower bound bivariate distributions between f & b
                            covLo = covLo + 0.5*(fCdf(i)+bCdf(j)-1+abs(fCdf(i)+bCdf(j)-1)) - fCdf(i).*bCdf(j);
                        end
                    end
                    corrLo = covLo/(fSd*bSd); % Lower correlation between f & b
                    theta = c/corrLo;
                end

                jpdfLo = 0.5*(fCdf(1)+bCdf(1)-1+abs(fCdf(1)+bCdf(1)-1));
                jpdf = theta.*jpdfLo + (1-theta).*fPdf(1).*bPdf(1);
                if jpdf~=0
                    jointEntropy = real(-jpdf.*log2(jpdf)); % Joint entropy
                end

                % 1-D boundaries
                for i=2:l
                    jpdfLo = 0.5*(fCdf(i)+bCdf(1)-1+abs(fCdf(i)+bCdf(1)-1)) - ...
                        0.5*(fCdf(i-1)+bCdf(1)-1+abs(fCdf(i-1)+bCdf(1)-1));
                    jpdf = theta.*jpdfLo + (1-theta).*fPdf(i).*bPdf(1);
                    if jpdf~=0
                        jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                    end
                end
                for j=2:l
                    jpdfLo = 0.5*(fCdf(1)+bCdf(j)-1+abs(fCdf(1)+bCdf(j)-1)) - ...
                        0.5*(fCdf(1)+bCdf(j-1)-1+abs(fCdf(1)+bCdf(j-1)-1));
                    jpdf = theta.*jpdfLo + (1-theta).*fPdf(1).*bPdf(j);
                    if jpdf~=0
                        jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                    end
                end
                % 2-D walls
                for i=2:l
                    for j=2:l
                        jpdfLo = 0.5*(fCdf(i)+bCdf(j)-1+abs(fCdf(i)+bCdf(j)-1)) - ...
                            0.5*(fCdf(i-1)+bCdf(j)-1+abs(fCdf(i-1)+bCdf(j)-1)) - ...
                            0.5*(fCdf(i)+bCdf(j-1)-1+abs(fCdf(i)+bCdf(j-1)-1)) + ...
                            0.5*(fCdf(i-1)+bCdf(j-1)-1+abs(fCdf(i-1)+bCdf(j-1)-1));
                        jpdf = theta.*jpdfLo + (1-theta).*fPdf(i).*bPdf(j);
                        if jpdf~=0
                            jointEntropy = jointEntropy + real(-jpdf.*log2(jpdf)); % Joint entropy
                        end
                    end
                end

            end


            % Marginal entropies
            index = find(bPdf~=0);
            bEntropy = sum(-bPdf(index).*log2(bPdf(index)));
            index = find(fPdf~=0);
            fEntropy = sum(-fPdf(index).*log2(fPdf(index)));

            % Mutual information between b & f
            mi = bEntropy + fEntropy - jointEntropy;  

            % Overall normalized mutual information
            if mi == 0
                fmi_bf = 0;
            else
                fmi_bf = 2.*mi./(bEntropy+fEntropy);
            end

        end
end
