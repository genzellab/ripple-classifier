clear variables
cd('/media/adrian/6aa1794c-0320-4096-a7df-00ab0ba946dc/cbd_revision/renamed')
% Load cbd data
% Get all .mat files in the current directory that contain 'cbd' in their name
matFiles = dir('*cbd*.mat');
matFiles={matFiles.name}; % Get all cbd rats. 
Wave={};

for indx=1:length(matFiles)
   load(matFiles{indx})
   ratID = regexp(matFiles{indx}, 'ratID(\d+)_cbd', 'tokens');
   ratID=str2double(ratID{1});
%% Complex SWR
   varName = sprintf('GC_complex_swr_ratID%d_cbd', ratID);
   z = eval(varName);

for i=1:length(z.waveforms)
    
    trace=z.waveforms{i}(5,:); % 6 sec raw Pyr trace
    start_end=table2array([z.grouped_oscil_table(i,6:7)]);
    wave=trace(start_end(1):start_end(2));
    Wave=[Wave; {wave}];    
    
end

%% SWR
   varName = sprintf('GC_swr_ratID%d_cbd', ratID);
   z = eval(varName);

for i=1:length(z.waveforms)
    
    trace=z.waveforms{i}(5,:); % 6 sec raw Pyr trace
    start_end=table2array([z.grouped_oscil_table(i,6:7)]);
    wave=trace(start_end(1):start_end(2));
    Wave=[Wave; {wave}];    
    
end

%% Ripple
   varName = sprintf('GC_ripple_ratID%d_cbd', ratID);
   z = eval(varName);

for i=1:length(z.waveforms)
    
    trace=z.waveforms{i}(5,:); % 6 sec raw Pyr trace
    start_end=table2array([z.grouped_oscil_table(i,6:7)]);
    wave=trace(start_end(1):start_end(2));
    Wave=[Wave; {wave}];    
    
end

%Wave=Wave.';
    
end

%%
cd('/media/adrian/6aa1794c-0320-4096-a7df-00ab0ba946dc/cbd_revision')
rcbd=load('wave_cbd.mat');
rveh=load('wave_veh.mat');

rcbd=rcbd.Wave;
rveh=rveh.Wave;
%%
fn=600;
fs=fn;

Wn1=[50/(fn/2) 250/(fn/2)]; % Cutoff=100-300 Hz
% Wn1=[50/(fn/2) 80/(fn/2)]; 
[b1,a1] = butter(3,Wn1,'bandpass'); %Filter coefficients

Rcbd=cellfun(@(equis) filtfilt(b1,a1,equis), rcbd ,'UniformOutput',false);
Rveh=cellfun(@(equis) filtfilt(b1,a1,equis), rveh ,'UniformOutput',false);


freq_Rcbd=cellfun(@(equis) (meanfreq(equis,fs)) ,Rcbd,'UniformOutput',false);
freq_Rveh=cellfun(@(equis) (meanfreq(equis,fs)) ,Rveh,'UniformOutput',false);

freq_Rcbd=cell2mat(freq_Rcbd);
freq_Rveh=cell2mat(freq_Rveh);
%%

%%
bin_edges = linspace(50, 250, 60);

% Plot the histograms as probability density histograms
figure;
histogram(freq_Rcbd, bin_edges, 'Normalization', 'probability', 'FaceAlpha', 0.6);
hold on;
histogram(freq_Rveh, bin_edges, 'Normalization', 'probability', 'FaceAlpha', 0.6);

% Add labels and title for clarity
xlabel('Frequency (Hz)');
ylabel('Probability');
title('Histogram of Frequency');
legend('CBD', 'Veh');
hold off;
%%
shuffled_v=freq_Rveh(randperm(length(freq_Rveh)));
shuffled_cbd=freq_Rcbd(randperm(length(freq_Rcbd)));
shuffled_cbd=shuffled_cbd(1:6933);
%%
violin2(interleave2( shuffled_v, shuffled_cbd,'col'))
h=gca;
h.Title.String='Ripple Frequency'
h.Title.FontSize=16;
h.XAxis.FontSize=14;
h.YAxis.FontSize=14;

h.Children(3).FaceColor=[ 0 0.5 0 ]
h.Children(6).FaceColor=[0.5 0.5 0.5];
h.XTick=[1 2]
h.XTickLabel=[{'Veh'} {'CBD'}]

ylabel('Frequency (Hz)')
