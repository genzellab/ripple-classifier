
%% Vehicle
clc
clear
addpath('/home/genzellab/Desktop/Pelin/multipleDataInCell/unfiltered-HPC');

rats_veh=[3 4 9 201 203 206 210 211 213];

for rat_no=1:size(rats_veh,2)
    rat=rats_veh(rat_no);

    temp_var = strcat( "load('GC_ratID",num2str(rat),"_veh.mat');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "load('ratID",num2str(rat),"_HPC_veh.mat');");
    eval(sprintf('%s',temp_var));

    %% HPCpyra & HPCbelo - unfiltered 
    % Ripple
    temp_var = strcat( "for i=1:size(GC_ripple_waveforms_HPCpyra_ratID",num2str(rat),...
        ",1); GC_ripple_ratID",num2str(rat),"_veh.waveforms{i,1}(5,:)=GC_ripple_waveforms_HPCpyra_ratID",...
        num2str(rat),"{i,1}(1,:); GC_ripple_ratID",num2str(rat),"_veh.waveforms{i,1}(6,:)=GC_ripple_waveforms_HPCbelo_ratID",...
        num2str(rat),"{i,1}(1,:); end");
    eval(sprintf('%s',temp_var));

    % SW
    temp_var = strcat( "for i=1:size(GC_sw_waveforms_HPCpyra_ratID",num2str(rat),...
        ",1); GC_sw_ratID",num2str(rat),"_veh.waveforms{i,1}(5,:)=GC_sw_waveforms_HPCpyra_ratID",...
        num2str(rat),"{i,1}(1,:); GC_sw_ratID",num2str(rat),"_veh.waveforms{i,1}(6,:)=GC_sw_waveforms_HPCbelo_ratID",...
        num2str(rat),"{i,1}(1,:); end");
    eval(sprintf('%s',temp_var));

    % SWR
    temp_var = strcat( "for i=1:size(GC_swr_waveforms_HPCpyra_ratID",num2str(rat),...
        ",1); GC_swr_ratID",num2str(rat),"_veh.waveforms{i,1}(5,:)=GC_swr_waveforms_HPCpyra_ratID",...
        num2str(rat),"{i,1}(1,:); GC_swr_ratID",num2str(rat),"_veh.waveforms{i,1}(6,:)=GC_swr_waveforms_HPCbelo_ratID",...
        num2str(rat),"{i,1}(1,:); end");
    eval(sprintf('%s',temp_var));

    % cSWR
    temp_var = strcat( "for i=1:size(GC_complex_swr_waveforms_HPCpyra_ratID",num2str(rat),...
        ",1); GC_complex_swr_ratID",num2str(rat),"_veh.waveforms{i,1}(5,:)=GC_complex_swr_waveforms_HPCpyra_ratID",...
        num2str(rat),"{i,1}(1,:); GC_complex_swr_ratID",num2str(rat),"_veh.waveforms{i,1}(6,:)=GC_complex_swr_waveforms_HPCbelo_ratID",...
        num2str(rat),"{i,1}(1,:); end");
    eval(sprintf('%s',temp_var));


    data_info='1)PFCshal; 2)HPCpyra; 3)HPCbelo; 4)PFCdeep; 5)HPCpyra-unfiltered; 6)HPCbelo-unfiltered';

    temp_var = strcat( "save FT_ratID",num2str(rat),"_veh.mat GC_ripple_ratID",num2str(rat),...
        "_veh GC_sw_ratID",num2str(rat),"_veh GC_swr_ratID",num2str(rat),"_veh GC_complex_swr_ratID",num2str(rat),"_veh data_info");
    eval(sprintf('%s',temp_var));
    
    clearvars -except rat rats_veh rat_no
end


%% CBD
clc
clear

rats_cbd=[2, 5, 10, 11, 204, 205, 207, 209, 212, 214];

for rat_no=1:size(rats_cbd,2)
    rat=rats_cbd(rat_no);

    temp_var = strcat( "load('GC_ratID",num2str(rat),"_cbd.mat');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "load('ratID",num2str(rat),"_HPC_cbd.mat');");
    eval(sprintf('%s',temp_var));

    %% HPCpyra & HPCbelo - unfiltered 
    % Ripple
    temp_var = strcat( "for i=1:size(GC_ripple_waveforms_HPCpyra_ratID",num2str(rat),...
        ",1); GC_ripple_ratID",num2str(rat),"_cbd.waveforms{i,1}(5,:)=GC_ripple_waveforms_HPCpyra_ratID",...
        num2str(rat),"{i,1}(1,:); GC_ripple_ratID",num2str(rat),"_cbd.waveforms{i,1}(6,:)=GC_ripple_waveforms_HPCbelo_ratID",...
        num2str(rat),"{i,1}(1,:); end");
    eval(sprintf('%s',temp_var));

    % SW
    temp_var = strcat( "for i=1:size(GC_sw_waveforms_HPCpyra_ratID",num2str(rat),...
        ",1); GC_sw_ratID",num2str(rat),"_cbd.waveforms{i,1}(5,:)=GC_sw_waveforms_HPCpyra_ratID",...
        num2str(rat),"{i,1}(1,:); GC_sw_ratID",num2str(rat),"_cbd.waveforms{i,1}(6,:)=GC_sw_waveforms_HPCbelo_ratID",...
        num2str(rat),"{i,1}(1,:); end");
    eval(sprintf('%s',temp_var));

    % SWR
    temp_var = strcat( "for i=1:size(GC_swr_waveforms_HPCpyra_ratID",num2str(rat),...
        ",1); GC_swr_ratID",num2str(rat),"_cbd.waveforms{i,1}(5,:)=GC_swr_waveforms_HPCpyra_ratID",...
        num2str(rat),"{i,1}(1,:); GC_swr_ratID",num2str(rat),"_cbd.waveforms{i,1}(6,:)=GC_swr_waveforms_HPCbelo_ratID",...
        num2str(rat),"{i,1}(1,:); end");
    eval(sprintf('%s',temp_var));

    % cSWR
    temp_var = strcat( "for i=1:size(GC_complex_swr_waveforms_HPCpyra_ratID",num2str(rat),...
        ",1); GC_complex_swr_ratID",num2str(rat),"_cbd.waveforms{i,1}(5,:)=GC_complex_swr_waveforms_HPCpyra_ratID",...
        num2str(rat),"{i,1}(1,:); GC_complex_swr_ratID",num2str(rat),"_cbd.waveforms{i,1}(6,:)=GC_complex_swr_waveforms_HPCbelo_ratID",...
        num2str(rat),"{i,1}(1,:); end");
    eval(sprintf('%s',temp_var));


    data_info='1)PFCshal; 2)HPCpyra; 3)HPCbelo; 4)PFCdeep; 5)HPCpyra-unfiltered; 6)HPCbelo-unfiltered';

    temp_var = strcat( "save FT_ratID",num2str(rat),"_cbd.mat GC_ripple_ratID",num2str(rat),...
        "_cbd GC_sw_ratID",num2str(rat),"_cbd GC_swr_ratID",num2str(rat),"_cbd GC_complex_swr_ratID",num2str(rat),"_cbd data_info");
    eval(sprintf('%s',temp_var));
    
    clearvars -except rat rats_cbd rat_no
end


