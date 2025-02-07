% ���������
baseFolder = 'EHGs'; % ������� ������� ��� �������� �� ������
categories = {'preterm', 'term'}; % ���������� ���������
numSignals = 20; % ������� ������� ��� �� ����������

% ������������ ����������� �������������
results = struct();
figure;
hold on;

for c = 1:length(categories)
    category = categories{c};
    folderPath = fullfile(baseFolder, category);
    
    % �������� ���� ��� .txt �������
    dataFiles = dir(fullfile(folderPath, '*.txt'));
    numFiles = min(numSignals, length(dataFiles));
    
    sampleEntropies = zeros(1, numFiles);
    
    for i = 1:numFiles
        % ������� ���������
        filename = fullfile(dataFiles(i).folder, dataFiles(i).name);
        signalData = load(filename); % Alternative: dlmread(filename);
        
        % ������� ��� 4�� ������ (������ S1)
        S1 = signalData(:,4);
        
        % ����������� Sample Entropy
        sampleEntropies(i) = SampEn(2, 0.2*std(S1), S1);
    end
    
    % ���������� �������������
    results.(category) = sampleEntropies;
    
    % ������������
    bar((1:numFiles) + (c-1)*numFiles, sampleEntropies, 'DisplayName', category);
end

hold off;
xlabel('������� �������');
ylabel('Sample Entropy');
title('Sample Entropy ��� Preterm ��� Term ������');
legend;
grid on;

% �������� �������������
disp('Sample Entropy - Preterm Signals:');
disp(results.preterm);
disp('Sample Entropy - Term Signals:');
disp(results.term);
