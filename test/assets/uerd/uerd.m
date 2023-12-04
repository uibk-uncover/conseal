function [ S_STRUCT , pChange , ChangeRate ] = uerd(C_STRUCT, payload, seed)
% -------------------------------------------------------------------------
% UERD Embedding       |      August 2020       |      version 0.2
% -------------------------------------------------------------------------
% INPUT:
%  - C_STRUCT    - Struct representing JPEG compressed image (or path to JPEG file)
%  - Payload     - Embedding payload in bits per non-zeros AC DCT coefs (bpnzAC).
% OUTPUT:
%  - S_STRUCT    - Resulting stego jpeg STRUCT with embedded payload
%  - pChange     - Embedding change probabilities.
%  - ChangeRate  - Average number of changed pixels
% -------------------------------------------------------------------------
% License free implementation of the UERD embedding scheme
% Rémi Cogranne, UTT (Troyes University of Technology)
% All Rights Reserved.
% -------------------------------------------------------------------------
% This code is provided by the author under Creative Common License
% (CC BY-NC-SA 4.0) which, as explained on this webpage
% https://creativecommons.org/licenses/by-nc-sa/4.0/
% Allows modification, redistribution, provided that:
% * You share your code under the same license ;
% * That you give credits to the authors ;
% * The code is used only for non-commercial purposes (which includes
% education and research)
% -------------------------------------------------------------------------

% Read the JPEG image if needed
if ischar( C_STRUCT )
    C_STRUCT = jpeg_read( C_STRUCT );
end

DCT_coefs =  C_STRUCT.coef_arrays{1};
Quant_table = C_STRUCT.quant_tables{1};

wetConst = 10^13;
nzAC = nnz(DCT_coefs)-nnz(DCT_coefs(1:8:end,1:8:end));

% Beware, version 0.1 uses this padding which is not not correct
% (it does not take into account arrangement of DCT coefficients)
%DCT_coefs_padded = padarray(DCT_coefs, [8 8] , 'symmetric');
DCT_coefs_padded = zeros(size( DCT_coefs ) + 16);
DCT_coefs_padded(9:end-8, 9:end-8) = DCT_coefs(:,:); % Copy center
DCT_coefs_padded(1:8, :) = DCT_coefs_padded(9:16, :); % Copy top 8 rows
DCT_coefs_padded(end-7:end, :) = DCT_coefs_padded(end-15:end-8, :); % Copy bottom 8 rows
DCT_coefs_padded( : , 1:8, :) = DCT_coefs_padded( : , 9:16); % Copy 8 left columns
DCT_coefs_padded( : , end-7:end, :) = DCT_coefs_padded( : , end-15:end-8); % Copy 8 right columns

fun=@(x) sum( sum( abs(x) .* Quant_table ) ) - abs(x(1,1) ) .* Quant_table(1,1);
X = blkproc(  DCT_coefs_padded, [8 8], fun );
Divisor = conv2(X, 1/4 * [1 1 1 ; 1 4 1 ; 1 1 1 ], 'valid');
sizeX = size(Divisor);
rho = zeros(size(DCT_coefs));


for index_DCT1=1:sizeX(1)
    for index_DCT2=1:sizeX(2)
        rho( (index_DCT1-1)*8+1: index_DCT1*8 , (index_DCT2-1)*8+1: index_DCT2*8 ) = Quant_table / Divisor(index_DCT1,index_DCT2) ;
    end,
end,
i=1;j=1;
for index_DCT1=1:sizeX(1)
    for index_DCT2=1:sizeX(2)
        rho(i+(index_DCT1-1)*8 , j+(index_DCT2-1)*8) = 0.5*( Quant_table(i+1,j) + Quant_table(i,j+1) ) / Divisor(index_DCT1,index_DCT2);
    end,
end,


rho(rho==Inf) = wetConst;
rho(isnan(rho)) = wetConst;

rhoP1 = rho;
rhoM1 = rhoP1;
rhoP1(DCT_coefs>=1023) = wetConst;
rhoM1(DCT_coefs<=-1023) = wetConst;


%% Embedding simulation
[S_COEFFS pChangeP1 pChangeM1] = EmbeddingSimulator(DCT_coefs, rhoP1, rhoM1, round(payload * nzAC), seed);

S_STRUCT = C_STRUCT;
S_STRUCT.coef_arrays{1} = S_COEFFS;
S_STRUCT.optimize_coding = 1;
pChange = pChangeP1 + pChangeM1;
ChangeRate = sum(pChange(:))/numel(DCT_coefs); % Computing the change rate

end


function [ y pChangeP1 pChangeM1 ] = EmbeddingSimulator(x, rhoP1, rhoM1, m, seed)

    x = double(x);
    n = numel(x);

    lambda = calc_lambda(rhoP1, rhoM1, m, n);
    pChangeP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
    pChangeM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));

    RandStream.setGlobalStream(RandStream('mt19937ar','Seed', seed));
    % Flip the size such that transpose later restores the correct
    % dimensions
    randChange = rand(flip(size(x)));
    % Transpose randChange in order to match Python implementation
    randChange = transpose(randChange);
    y = x;
    y(randChange < pChangeP1) = y(randChange < pChangeP1) + 1;
    y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) = y(randChange >= pChangeP1 & randChange < pChangeP1+pChangeM1) - 1;

    function lambda = calc_lambda(rhoP1, rhoM1, message_length, n)

        l3 = 1e+3;
        m3 = double(message_length + 1);
        iterations = 0;
        while m3 > message_length
            l3 = l3 * 2;
            pP1 = (exp(-l3 .* rhoP1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            pM1 = (exp(-l3 .* rhoM1))./(1 + exp(-l3 .* rhoP1) + exp(-l3 .* rhoM1));
            m3 = ternary_entropyf(pP1, pM1);
            iterations = iterations + 1;
            if (iterations > 10)
                lambda = l3;
                return;
            end
        end

        l1 = 0;
        m1 = double(n);
        lambda = 0;

        alpha = double(message_length)/n;
        % limit search to 30 iterations
        % and require that relative payload embedded is roughly within 1/1000 of the required relative payload
        while  (double(m1-m3)/n > alpha/1000.0 ) && (iterations<30)
            lambda = l1+(l3-l1)/2;
            pP1 = (exp(-lambda .* rhoP1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            pM1 = (exp(-lambda .* rhoM1))./(1 + exp(-lambda .* rhoP1) + exp(-lambda .* rhoM1));
            m2 = ternary_entropyf(pP1, pM1);
    		if m2 < message_length
    			l3 = lambda;
    			m3 = m2;
            else
    			l1 = lambda;
    			m1 = m2;
            end
    		iterations = iterations + 1;
        end
    end

    function Ht = ternary_entropyf(pP1, pM1)
        pP1 = pP1(:);
        pM1 = pM1(:);
        Ht = -(pP1.*log2(pP1))-(pM1.*log2(pM1))-((1-pP1-pM1).*log2(1-pP1-pM1));
        Ht(isnan(Ht)) = 0;
        Ht = sum(Ht);
    end

end

