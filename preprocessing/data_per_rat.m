
% SWR
x=strcat('GC_swr_ratID3_veh','.waveforms')
x=eval([x]);
length(x)
for i=1: length(x)
    HPCpyra_swr_veh(i,:)=x{i}(2,:);
    PFCshal_swr_veh(i,:)=x{i}(1,:);
    PFCdeep_swr_veh(i,:)=x{i}(4,:);
end


% RIPPLE
x=strcat('GC_ripple_ratID3_veh','.waveforms')
x=eval([x]);
length(x)
for i=1: length(x)
    HPCpyra_ripple_veh(i,:)=x{i}(2,:);
    PFCshal_ripple_veh(i,:)=x{i}(1,:);
    PFCdeep_ripple_veh(i,:)=x{i}(4,:);
end


% Complex swr
x=strcat('GC_complex_swr_ratID3_veh','.waveforms')
x=eval([x]);
length(x)
for i=1: length(x)
    HPCpyra_complex_swr_veh(i,:)=x{i}(2,:);
    PFCshal_complex_swr_veh(i,:)=x{i}(1,:);
    PFCdeep_complex_swr_veh(i,:)=x{i}(4,:);    
end
%%
clearvars -except  HPCpyra_complex_swr_veh HPCpyra_ripple_veh HPCpyra_swr_veh

save('HPCpyra_events_ratID3')

%% PFC shal
clearvars -except  PFCshal_complex_swr_veh PFCshal_ripple_veh PFCshal_swr_veh

save('PFCshal_events_ratID3')
%% PFC deep
clearvars -except  PFCdeep_complex_swr_veh PFCdeep_ripple_veh PFCdeep_swr_veh

save('PFCdeep_events_ratID3')

%%
clear variables