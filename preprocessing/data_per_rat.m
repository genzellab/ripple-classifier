
% SWR
x=strcat('GC_swr_ratID214_cbd','.waveforms')
x=eval([x]);
length(x)
for i=1: length(x)
    HPCpyra_swr_cbd(i,:)=x{i}(2,:);
    PFCshal_swr_cbd(i,:)=x{i}(1,:);
    PFCdeep_swr_cbd(i,:)=x{i}(4,:);
end


% RIPPLE
x=strcat('GC_ripple_ratID214_cbd','.waveforms')
x=eval([x]);
length(x)
for i=1: length(x)
    HPCpyra_ripple_cbd(i,:)=x{i}(2,:);
    PFCshal_ripple_cbd(i,:)=x{i}(1,:);
    PFCdeep_ripple_cbd(i,:)=x{i}(4,:);
end


% Complex swr
x=strcat('GC_complex_swr_ratID214_cbd','.waveforms')
x=eval([x]);
length(x)
for i=1: length(x)
    HPCpyra_complex_swr_cbd(i,:)=x{i}(2,:);
    PFCshal_complex_swr_cbd(i,:)=x{i}(1,:);
    PFCdeep_complex_swr_cbd(i,:)=x{i}(4,:);    
end
%%
clearvars -except  HPCpyra_complex_swr_cbd HPCpyra_ripple_cbd HPCpyra_swr_cbd

save('HPCpyra_events_ratID214')

%% PFC shal
clearvars -except  PFCshal_complex_swr_cbd PFCshal_ripple_cbd PFCshal_swr_cbd

save('PFCshal_events_ratID214')
%% PFC deep
clearvars -except  PFCdeep_complex_swr_cbd PFCdeep_ripple_cbd PFCdeep_swr_cbd

save('PFCdeep_events_ratID214')

%%
clear variables