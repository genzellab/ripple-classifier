

%% Unfiltered and Filtered data have different numbers of events due to artifact removal.
%% Remove extra events from unfiltered.
clc
clear
cd /home/genzellab/Desktop/Pelin/multipleDataInCell

%%%%%%%%%%%%%
%% Vehicle %%
%%%%%%%%%%%%%

rats_veh=[3 4 9 201 203 206 210 211 213];
for rat_no=1:size(rats_veh,2) % rat_no=7
    rat=rats_veh(rat_no);
    rat

    %% Ripple
    cd /home/genzellab/Desktop/Pelin/multipleDataInCell/unfiltered-HPC

    temp_var = strcat( "load('ratID",num2str(rat),"_HPC_veh.mat');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "unfiltered=table2array(ripple_grouped_oscil_table_HPCpyra_ratID",...
        num2str(rat),"(:,4));");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "unfiltered_wave_HPCpyra=GC_ripple_waveforms_HPCpyra_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( " unfiltered_wave_HPCbelo=GC_ripple_waveforms_HPCbelo_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    cd /home/genzellab/Desktop/Pelin/multipleDataInCell

    temp_var = strcat( " load('FT_ratID",num2str(rat),"_veh.mat');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "filtered=table2array(GC_ripple_ratID",num2str(rat),...
        "_veh.grouped_oscil_table(:,4));");
    eval(sprintf('%s',temp_var));

    [nonsharedvals,idx] = setxor(unfiltered,filtered,'stable');

    unfiltered_wave_HPCpyra(idx)=[  ] ;
    unfiltered_wave_HPCbelo(idx)=[  ] ;

    temp_var = strcat( "for i=1:size(unfiltered_wave_HPCpyra,1); GC_ripple_ratID",num2str(rat),...
        "_veh.waveforms{i,1}(5,:)=unfiltered_wave_HPCpyra{i,1}(1,:); GC_ripple_ratID",num2str(rat),...
        "_veh.waveforms{i,1}(6,:)=unfiltered_wave_HPCbelo{i,1}(1,:);end");
    eval(sprintf('%s',temp_var));


    %hnd=size(GC_ripple_ratID3_veh.waveforms,1);

    temp_var = strcat("GC_ripple_ratID",num2str(rat),"_veh.waveforms(end-size(idx,1)+1:end)=[ ] ;");
    eval(sprintf('%s',temp_var));


    %% SW



    temp_var = strcat( "unfiltered=table2array(sw_grouped_oscil_table_HPCpyra_ratID",...
        num2str(rat),"(:,4));");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "unfiltered_wave_HPCpyra=GC_sw_waveforms_HPCpyra_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( " unfiltered_wave_HPCbelo=GC_sw_waveforms_HPCbelo_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "filtered=table2array(GC_sw_ratID",num2str(rat),...
        "_veh.grouped_oscil_table(:,4));");
    eval(sprintf('%s',temp_var));

    [nonsharedvals,idx] = setxor(unfiltered,filtered,'stable');

    unfiltered_wave_HPCpyra(idx)=[  ] ;
    unfiltered_wave_HPCbelo(idx)=[  ] ;

    temp_var = strcat( "for i=1:size(unfiltered_wave_HPCpyra,1); GC_sw_ratID",num2str(rat),...
        "_veh.waveforms{i,1}(5,:)=unfiltered_wave_HPCpyra{i,1}(1,:); GC_sw_ratID",num2str(rat),...
        "_veh.waveforms{i,1}(6,:)=unfiltered_wave_HPCbelo{i,1}(1,:);end");
    eval(sprintf('%s',temp_var));


    %hnd=size(GC_sw_ratID3_veh.waveforms,1);

    temp_var = strcat( "GC_sw_ratID",num2str(rat),"_veh.waveforms(end-size(idx,1)+1:end)=[ ] ;");
    eval(sprintf('%s',temp_var));





    %% SWR
    temp_var = strcat( "unfiltered=table2array(swr_grouped_oscil_table_HPCpyra_ratID",...
        num2str(rat),"(:,4));");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "unfiltered_wave_HPCpyra=GC_swr_waveforms_HPCpyra_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( " unfiltered_wave_HPCbelo=GC_swr_waveforms_HPCbelo_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "filtered=table2array(GC_swr_ratID",num2str(rat),...
        "_veh.grouped_oscil_table(:,4));");
    eval(sprintf('%s',temp_var));

    [nonsharedvals,idx] = setxor(unfiltered,filtered,'stable');

    unfiltered_wave_HPCpyra(idx)=[  ] ;
    unfiltered_wave_HPCbelo(idx)=[  ] ;

    temp_var = strcat( "for i=1:size(unfiltered_wave_HPCpyra,1); GC_swr_ratID",num2str(rat),...
        "_veh.waveforms{i,1}(5,:)=unfiltered_wave_HPCpyra{i,1}(1,:); GC_swr_ratID",num2str(rat),...
        "_veh.waveforms{i,1}(6,:)=unfiltered_wave_HPCbelo{i,1}(1,:);end");
    eval(sprintf('%s',temp_var));


    %hnd=size(GC_swr_ratID3_veh.waveforms,1);

    temp_var = strcat( "GC_swr_ratID",num2str(rat),"_veh.waveforms(end-size(idx,1)+1:end)=[ ] ;");
    eval(sprintf('%s',temp_var));




    %% Complex SWR
    temp_var = strcat( "unfiltered=table2array(complex_swr_grouped_oscil_table_HPCpyra_ratID",...
        num2str(rat),"(:,4));");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "unfiltered_wave_HPCpyra=GC_complex_swr_waveforms_HPCpyra_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( " unfiltered_wave_HPCbelo=GC_complex_swr_waveforms_HPCbelo_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "filtered=table2array(GC_complex_swr_ratID",num2str(rat),...
        "_veh.grouped_oscil_table(:,4));");
    eval(sprintf('%s',temp_var));

    [nonsharedvals,idx] = setxor(unfiltered,filtered,'stable');

    unfiltered_wave_HPCpyra(idx)=[  ] ;
    unfiltered_wave_HPCbelo(idx)=[  ] ;

    temp_var = strcat( "for i=1:size(unfiltered_wave_HPCpyra,1); GC_complex_swr_ratID",num2str(rat),...
        "_veh.waveforms{i,1}(5,:)=unfiltered_wave_HPCpyra{i,1}(1,:); GC_complex_swr_ratID",num2str(rat),...
        "_veh.waveforms{i,1}(6,:)=unfiltered_wave_HPCbelo{i,1}(1,:);end");
    eval(sprintf('%s',temp_var));


    %hnd=size(GC_complex_swr_ratID3_veh.waveforms,1);

    temp_var = strcat( "GC_complex_swr_ratID",num2str(rat),"_veh.waveforms(end-size(idx,1)+1:end)=[ ] ;");
    eval(sprintf('%s',temp_var));

    %% saving
    cd /home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT
    temp_var = strcat( "save FT_ratID",num2str(rat),"_veh.mat GC_ripple_ratID",num2str(rat),...
        "_veh GC_sw_ratID",num2str(rat),"_veh GC_swr_ratID",num2str(rat),...
        "_veh GC_complex_swr_ratID",num2str(rat),"_veh data_info");
    eval(sprintf('%s',temp_var));

    cd /home/genzellab/Desktop/Pelin/multipleDataInCell
    clearvars -except rat rats_veh rat_no

end

%%%%%%%%%
%% CBD %%
%%%%%%%%%
rats_cbd=[2, 5, 10, 11, 204, 205, 207, 209, 212, 214];
for rat_no=1:size(rats_cbd,2) % rat_no=7
    rat=rats_cbd(rat_no);
    rat

    %% Ripple
    cd /home/genzellab/Desktop/Pelin/multipleDataInCell/unfiltered-HPC

    temp_var = strcat( "load('ratID",num2str(rat),"_HPC_cbd.mat');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "unfiltered=table2array(ripple_grouped_oscil_table_HPCpyra_ratID",...
        num2str(rat),"(:,4));");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "unfiltered_wave_HPCpyra=GC_ripple_waveforms_HPCpyra_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( " unfiltered_wave_HPCbelo=GC_ripple_waveforms_HPCbelo_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    cd /home/genzellab/Desktop/Pelin/multipleDataInCell

    temp_var = strcat( " load('FT_ratID",num2str(rat),"_cbd.mat');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "filtered=table2array(GC_ripple_ratID",num2str(rat),...
        "_cbd.grouped_oscil_table(:,4));");
    eval(sprintf('%s',temp_var));

    [nonsharedvals,idx] = setxor(unfiltered,filtered,'stable');

    unfiltered_wave_HPCpyra(idx)=[  ] ;
    unfiltered_wave_HPCbelo(idx)=[  ] ;

    temp_var = strcat( "for i=1:size(unfiltered_wave_HPCpyra,1); GC_ripple_ratID",num2str(rat),...
        "_cbd.waveforms{i,1}(5,:)=unfiltered_wave_HPCpyra{i,1}(1,:); GC_ripple_ratID",num2str(rat),...
        "_cbd.waveforms{i,1}(6,:)=unfiltered_wave_HPCbelo{i,1}(1,:);end");
    eval(sprintf('%s',temp_var));


    %hnd=size(GC_ripple_ratID3_cbd.waveforms,1);

    temp_var = strcat("GC_ripple_ratID",num2str(rat),"_cbd.waveforms(end-size(idx,1)+1:end)=[ ] ;");
    eval(sprintf('%s',temp_var));


    %% SW



    temp_var = strcat( "unfiltered=table2array(sw_grouped_oscil_table_HPCpyra_ratID",...
        num2str(rat),"(:,4));");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "unfiltered_wave_HPCpyra=GC_sw_waveforms_HPCpyra_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( " unfiltered_wave_HPCbelo=GC_sw_waveforms_HPCbelo_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "filtered=table2array(GC_sw_ratID",num2str(rat),...
        "_cbd.grouped_oscil_table(:,4));");
    eval(sprintf('%s',temp_var));

    [nonsharedvals,idx] = setxor(unfiltered,filtered,'stable');

    unfiltered_wave_HPCpyra(idx)=[  ] ;
    unfiltered_wave_HPCbelo(idx)=[  ] ;

    temp_var = strcat( "for i=1:size(unfiltered_wave_HPCpyra,1); GC_sw_ratID",num2str(rat),...
        "_cbd.waveforms{i,1}(5,:)=unfiltered_wave_HPCpyra{i,1}(1,:); GC_sw_ratID",num2str(rat),...
        "_cbd.waveforms{i,1}(6,:)=unfiltered_wave_HPCbelo{i,1}(1,:);end");
    eval(sprintf('%s',temp_var));


    %hnd=size(GC_sw_ratID3_cbd.waveforms,1);

    temp_var = strcat( "GC_sw_ratID",num2str(rat),"_cbd.waveforms(end-size(idx,1)+1:end)=[ ] ;");
    eval(sprintf('%s',temp_var));





    %% SWR
    temp_var = strcat( "unfiltered=table2array(swr_grouped_oscil_table_HPCpyra_ratID",...
        num2str(rat),"(:,4));");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "unfiltered_wave_HPCpyra=GC_swr_waveforms_HPCpyra_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( " unfiltered_wave_HPCbelo=GC_swr_waveforms_HPCbelo_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "filtered=table2array(GC_swr_ratID",num2str(rat),...
        "_cbd.grouped_oscil_table(:,4));");
    eval(sprintf('%s',temp_var));

    [nonsharedvals,idx] = setxor(unfiltered,filtered,'stable');

    unfiltered_wave_HPCpyra(idx)=[  ] ;
    unfiltered_wave_HPCbelo(idx)=[  ] ;

    temp_var = strcat( "for i=1:size(unfiltered_wave_HPCpyra,1); GC_swr_ratID",num2str(rat),...
        "_cbd.waveforms{i,1}(5,:)=unfiltered_wave_HPCpyra{i,1}(1,:); GC_swr_ratID",num2str(rat),...
        "_cbd.waveforms{i,1}(6,:)=unfiltered_wave_HPCbelo{i,1}(1,:);end");
    eval(sprintf('%s',temp_var));


    %hnd=size(GC_swr_ratID3_cbd.waveforms,1);

    temp_var = strcat( "GC_swr_ratID",num2str(rat),"_cbd.waveforms(end-size(idx,1)+1:end)=[ ] ;");
    eval(sprintf('%s',temp_var));




    %% Complex SWR
    temp_var = strcat( "unfiltered=table2array(complex_swr_grouped_oscil_table_HPCpyra_ratID",...
        num2str(rat),"(:,4));");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "unfiltered_wave_HPCpyra=GC_complex_swr_waveforms_HPCpyra_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( " unfiltered_wave_HPCbelo=GC_complex_swr_waveforms_HPCbelo_ratID",...
        num2str(rat),";");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "filtered=table2array(GC_complex_swr_ratID",num2str(rat),...
        "_cbd.grouped_oscil_table(:,4));");
    eval(sprintf('%s',temp_var));

    [nonsharedvals,idx] = setxor(unfiltered,filtered,'stable');

    unfiltered_wave_HPCpyra(idx)=[  ] ;
    unfiltered_wave_HPCbelo(idx)=[  ] ;

    temp_var = strcat( "for i=1:size(unfiltered_wave_HPCpyra,1); GC_complex_swr_ratID",num2str(rat),...
        "_cbd.waveforms{i,1}(5,:)=unfiltered_wave_HPCpyra{i,1}(1,:); GC_complex_swr_ratID",num2str(rat),...
        "_cbd.waveforms{i,1}(6,:)=unfiltered_wave_HPCbelo{i,1}(1,:);end");
    eval(sprintf('%s',temp_var));


    %hnd=size(GC_complex_swr_ratID3_cbd.waveforms,1);

    temp_var = strcat( "GC_complex_swr_ratID",num2str(rat),"_cbd.waveforms(end-size(idx,1)+1:end)=[ ] ;");
    eval(sprintf('%s',temp_var));

    %% saving
    cd /home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT
    temp_var = strcat( "save FT_ratID",num2str(rat),"_cbd.mat GC_ripple_ratID",num2str(rat),...
        "_cbd GC_sw_ratID",num2str(rat),"_cbd GC_swr_ratID",num2str(rat),...
        "_cbd GC_complex_swr_ratID",num2str(rat),"_cbd data_info");
    eval(sprintf('%s',temp_var));

    cd /home/genzellab/Desktop/Pelin/multipleDataInCell
    clearvars -except rat rats_cbd rat_no

end
clear
cd /home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT



