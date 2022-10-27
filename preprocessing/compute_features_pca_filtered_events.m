cd('/home/adrian/Documents/SWR_classifier/wetransfer_per_rat/Data_for_Ripple_Classifier/cbd')
load('GC_ratID2_cbd.mat')

z=GC_complex_swr_ratID2_cbd;

%%
%PCA data
cd('/home/adrian/Documents/SWR_classifier/wetransfer_per_rat/features_pca_tranformed/HPCpyra_PCA_VEH_test_CBD')

load('PCA_HPCpyra_events_ratID2.mat')
Wave={};
for i=1:length(z.waveforms)
    i
    trace=HPCpyra_complex_swr_cbd(i,:);
    start_end=table2array([z.grouped_oscil_table(i,6:7)]);
    wave=trace(start_end(1):start_end(2));
    Wave=[Wave; {wave}];    
    
end

Wave=Wave.';

[~,~,~,~,~,~,~,~,~,~,PCA_features]=event_characteristics(Wave,0,0,600);
PCA_features=PCA_features(:,2:end);

x.ripple_trace=Wave';
x.PCA_features=PCA_features;
xo
%%
cd('/home/adrian/Documents/SWR_classifier/wetransfer_per_rat/Data_for_Ripple_Classifier/cbd/new_GC_features_entropy')
%Now overwrite GC_ratID214_cbd.mat with x
GC_ripple_ratID214_cbd=x;
GC_swr_ratID214_cbd=y;
GC_complex_swr_ratID214_cbd=z;

clearvars -except -regexp GC data_info
save GC_ratID214_cbd.mat
