function e = SampEn(m, r, data)
% SampEn - Computes the Sample Entropy of a time series
% m: embedding dimension
% r: tolerance (usually 0.2 * std(data))
% data: input time series

N = length(data); % Length of the time series
count = zeros(1,2);
phi = zeros(1,2);

for j = 1:2
    m_j = m + j - 1;
    tempMatrix = zeros(N-m_j+1, m_j);
    
    for i = 1:(N-m_j+1)
        tempMatrix(i,:) = data(i:i+m_j-1);
    end
    
    for i = 1:(N-m_j+1)
        dist = max(abs(tempMatrix - tempMatrix(i,:)), [], 2);
        count(j) = count(j) + sum(dist < r) - 1; % Exclude self-matches
    end
    
    phi(j) = count(j) / (N - m_j + 1);
end

if phi(1) == 0 || phi(2) == 0
    e = NaN; % Avoid log(0)
else
    e = -log(phi(2) / phi(1));
end
end
