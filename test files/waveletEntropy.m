clc; clear; close all;

% Ορισμός φακέλων δεδομένων
preterm_folder = 'EHGs/preterm/';
term_folder = 'EHGs/term/';

% Λίστα αρχείων
preterm_files = dir(fullfile(preterm_folder, '*.txt'));
term_files = dir(fullfile(term_folder, '*.txt'));

% Επιλογή 10 αρχείων από κάθε κατηγορία
num_files = 10;
preterm_files = preterm_files(1:num_files);
term_files = term_files(1:num_files);

% Χωρητικότητα για αποθήκευση εντροπιών
sampen_preterm = zeros(1, num_files);
sampen_term = zeros(1, num_files);

% Παράμετροι Sample Entropy
m = 2;  % Μήκος προτύπου
r_factor = 0.2; % r ως ποσοστό της std

% Υπολογισμός εντροπίας για τα preterm σήματα
for i = 1:num_files
    file_path = fullfile(preterm_folder, preterm_files(i).name);
    signal = read_s1_column(file_path);
    
    % Υποδειγματοληψία (downsampling)
    signal = signal(1:10:end);
    
    % Υπολογισμός Sample Entropy
    sampen_preterm(i) = samp_entropy(signal, m, r_factor);
end

% Υπολογισμός εντροπίας για τα term σήματα
for i = 1:num_files
    file_path = fullfile(term_folder, term_files(i).name);
    signal = read_s1_column(file_path);
    
    % Υποδειγματοληψία (downsampling)
    signal = signal(1:10:end);
    
    % Υπολογισμός Sample Entropy
    sampen_term(i) = samp_entropy(signal, m, r_factor);
end

% Εμφάνιση αποτελεσμάτων
disp('Sample Entropy για Preterm:');
disp(sampen_preterm);
disp('Sample Entropy για Term:');
disp(sampen_term);

% Σύγκριση με boxplot
figure;
boxplot([sampen_preterm', sampen_term'], {'Preterm', 'Term'});
ylabel('Sample Entropy');
title('Comparison of Sample Entropy for Term & Preterm Signals');

% Υπολογισμός στατιστικής διαφοράς
[h, p] = ttest2(sampen_preterm, sampen_term);
disp(['p-value της σύγκρισης: ', num2str(p)]);

%% Συναρτήσεις (μεταφέρονται στο τέλος του script)
function s1_data = read_s1_column(file_path)
    data = readmatrix(file_path);
    s1_data = data(:, 4); % 4η στήλη (S1 κανάλι)
end

function SampEn = samp_entropy(signal, m, r_factor)
    N = length(signal);
    r = r_factor * std(signal); % Υπολογισμός r ως ποσοστό της std

    % Υπολογισμός phi(m) και phi(m+1)
    phi_m = mean(arrayfun(@(i) sum(abs(signal(i:i+m-1) - signal(i+1:i+m)) <= r) / (N - m), 1:N-m));
    phi_m1 = mean(arrayfun(@(i) sum(abs(signal(i:i+m) - signal(i+1:i+m+1)) <= r) / (N - m - 1), 1:N-m-1));

    % Υπολογισμός Sample Entropy
    if phi_m1 == 0
        SampEn = NaN; % Αποφυγή log(0)
    else
        SampEn = -log(phi_m1 / phi_m);
    end
end
