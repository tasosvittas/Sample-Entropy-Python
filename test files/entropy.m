% Ρυθμίσεις
baseFolder = 'EHGs'; % Βασικός φάκελος που περιέχει τα αρχεία
categories = {'preterm', 'term'}; % Κατηγορίες δεδομένων
numSignals = 20; % Αριθμός σημάτων που θα αναλύσουμε

% Προετοιμασία αποθήκευσης αποτελεσμάτων
results = struct();
figure;
hold on;

for c = 1:length(categories)
    category = categories{c};
    folderPath = fullfile(baseFolder, category);
    
    % Ανάγνωση όλων των .txt αρχείων
    dataFiles = dir(fullfile(folderPath, '*.txt'));
    numFiles = min(numSignals, length(dataFiles));
    
    sampleEntropies = zeros(1, numFiles);
    
    for i = 1:numFiles
        % Φόρτωση δεδομένων
        filename = fullfile(dataFiles(i).folder, dataFiles(i).name);
        signalData = load(filename); % Alternative: dlmread(filename);
        
        % Επιλογή της 4ης στήλης (κανάλι S1)
        S1 = signalData(:,4);
        
        % Υπολογισμός Sample Entropy
        sampleEntropies(i) = SampEn(2, 0.2*std(S1), S1);
    end
    
    % Αποθήκευση αποτελεσμάτων
    results.(category) = sampleEntropies;
    
    % Οπτικοποίηση
    bar((1:numFiles) + (c-1)*numFiles, sampleEntropies, 'DisplayName', category);
end

hold off;
xlabel('Αριθμός Σήματος');
ylabel('Sample Entropy');
title('Sample Entropy για Preterm και Term Σήματα');
legend;
grid on;

% Εμφάνιση Αποτελεσμάτων
disp('Sample Entropy - Preterm Signals:');
disp(results.preterm);
disp('Sample Entropy - Term Signals:');
disp(results.term);
