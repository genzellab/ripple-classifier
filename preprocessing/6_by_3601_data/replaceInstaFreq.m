

%%
clc
clear
cd("/home/genzellab/Desktop/Pelin/multipleDataInCell/final_FT");


%%%%%%%%%%%%%
%% VEHICLE %%
%%%%%%%%%%%%%
rats_veh=[3 4 9 201 203 206 210 211 213];

for rats=1:size(rats_veh,2)
    rat=rats_veh(rats);
    [ 'ratID',num2str(rat),' - Vehicle' ] 

    % convert traces from .mat files to .csv
    
     temp_var = strcat( "load FT_ratID",num2str(rat),"_veh.mat");
    eval(sprintf('%s',temp_var));

    
     temp_var = strcat( "GC_ripple=GC_ripple_ratID",num2str(rat),"_veh;");
    eval(sprintf('%s',temp_var));

    
     temp_var = strcat( "GC_sw=GC_sw_ratID",num2str(rat),"_veh;");
    eval(sprintf('%s',temp_var));

    
     temp_var = strcat( "GC_swr=GC_swr_ratID",num2str(rat),"_veh;");
    eval(sprintf('%s',temp_var));


     temp_var = strcat( "GC_complex_swr=GC_complex_swr_ratID",num2str(rat),"_veh;");
    eval(sprintf('%s',temp_var));


    %% Ripple
    wave_ripple_hpcbelo=[];
    for i=1:size(GC_ripple.waveforms,1)
        x=GC_ripple.waveforms{i,1}(3,:);
        wave_ripple_hpcbelo=[wave_ripple_hpcbelo; x];
    end
    save wave_ripple_hpcbelo.mat wave_ripple_hpcbelo
    FileData = load('wave_ripple_hpcbelo.mat');
    csvwrite('wave_ripple_hpcbelo.csv', FileData.wave_ripple_hpcbelo);

    clear FileData
    duration_ripple=[];
    for i=1:size(GC_ripple.grouped_oscil_table,1)
        x=table2array(GC_ripple.grouped_oscil_table(i,6:7));
        duration_ripple=[duration_ripple; x];
    end
    save duration_ripple.mat duration_ripple
    FileData = load('duration_ripple.mat');
    csvwrite('duration_ripple.csv', FileData.duration_ripple);

    %% SW
    wave_sw_hpcbelo=[];
    for i=1:size(GC_sw.waveforms,1)
        x=GC_sw.waveforms{i,1}(3,:);
        wave_sw_hpcbelo=[wave_sw_hpcbelo; x];
    end
    save wave_sw_hpcbelo.mat wave_sw_hpcbelo
    FileData = load('wave_sw_hpcbelo.mat');
    csvwrite('wave_sw_hpcbelo.csv', FileData.wave_sw_hpcbelo);

    clear FileData
    duration_sw=[];
    for i=1:size(GC_sw.grouped_oscil_table,1)
        x=table2array(GC_sw.grouped_oscil_table(i,6:7));
        duration_sw=[duration_sw; x];
    end
    save duration_sw.mat duration_sw
    FileData = load('duration_sw.mat');
    csvwrite('duration_sw.csv', FileData.duration_sw);

    %% SWR
    wave_swr_hpcbelo=[];
    for i=1:size(GC_swr.waveforms,1)
        x=GC_swr.waveforms{i,1}(3,:);
        wave_swr_hpcbelo=[wave_swr_hpcbelo; x];
    end
    save wave_swr_hpcbelo.mat wave_swr_hpcbelo
    FileData = load('wave_swr_hpcbelo.mat');
    csvwrite('wave_swr_hpcbelo.csv', FileData.wave_swr_hpcbelo);

    clear FileData
    duration_swr=[];
    for i=1:size(GC_swr.grouped_oscil_table,1)
        x=table2array(GC_swr.grouped_oscil_table(i,6:7));
        duration_swr=[duration_swr; x];
    end
    save duration_swr.mat duration_swr
    FileData = load('duration_swr.mat');
    csvwrite('duration_swr.csv', FileData.duration_swr);

    %% Complex SWR
    wave_complex_swr_hpcbelo=[];
    for i=1:size(GC_complex_swr.waveforms,1)
        x=GC_complex_swr.waveforms{i,1}(3,:);
        wave_complex_swr_hpcbelo=[wave_complex_swr_hpcbelo; x];
    end
    save wave_complex_swr_hpcbelo.mat wave_complex_swr_hpcbelo
    FileData = load('wave_complex_swr_hpcbelo.mat');
    csvwrite('wave_complex_swr_hpcbelo.csv', FileData.wave_complex_swr_hpcbelo);

    clear FileData
    duration_complex_swr=[];
    for i=1:size(GC_complex_swr.grouped_oscil_table,1)
        x=table2array(GC_complex_swr.grouped_oscil_table(i,6:7));
        duration_complex_swr=[duration_complex_swr; x];
    end
    save duration_complex_swr.mat duration_complex_swr
    FileData = load('duration_complex_swr.mat');
    csvwrite('duration_complex_swr.csv', FileData.duration_complex_swr);

    % PYTHON
    system('python compute-insta-freq.py');

    % features from .csv to .mat and replace
    features_ripple = readmatrix("features_ripple.csv");
    features_ripple=features_ripple(2:end,2);
    GC_ripple.HPCbelo_features(:,1)=features_ripple;

    features_sw = readmatrix("features_sw.csv");
    features_sw=features_sw(2:end,2);
    GC_sw.HPCbelo_features(:,1)=features_sw;

    features_swr = readmatrix("features_swr.csv");
    features_swr=features_swr(2:end,2);
    GC_swr.HPCbelo_features(:,1)=features_swr;

    features_complex_swr = readmatrix("features_complex_swr.csv");
    features_complex_swr=features_complex_swr(2:end,2);
    GC_complex_swr.HPCbelo_features(:,1)=features_complex_swr;
    
    
    %
    
     temp_var = strcat( "GC_ripple_ratID",num2str(rat),"_veh=GC_ripple;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_sw_ratID",num2str(rat),"_veh=GC_sw;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_swr_ratID",num2str(rat),"_veh=GC_swr;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_complex_swr_ratID",num2str(rat),"_veh=GC_complex_swr;");
    eval(sprintf('%s',temp_var));

    
    %
     temp_var = strcat( "save FTx_ratID",num2str(rat),"_veh.mat data_info GC_ripple_ratID",num2str(rat),"_veh GC_sw_ratID",num2str(rat),...
         "_veh GC_swr_ratID",num2str(rat),"_veh GC_complex_swr_ratID",num2str(rat),"_veh");
    eval(sprintf('%s',temp_var));

    clearvars -except rat rats_veh rats
    %delete 
end


%%%%%%%%
%% CBD %
%%%%%%%%
clc
clear variables
rats_cbd=[2 5 10 11 204 205 207 209 212 214];

for rats=1:size(rats_cbd,2)
    rat=rats_cbd(rats);
    [ 'ratID',num2str(rat),' - CBD' ] 

    % convert traces from .mat files to .csv
    
     temp_var = strcat( "load FT_ratID",num2str(rat),"_cbd.mat");
    eval(sprintf('%s',temp_var));

    
     temp_var = strcat( "GC_ripple=GC_ripple_ratID",num2str(rat),"_cbd;");
    eval(sprintf('%s',temp_var));

    
     temp_var = strcat( "GC_sw=GC_sw_ratID",num2str(rat),"_cbd;");
    eval(sprintf('%s',temp_var));

    
     temp_var = strcat( "GC_swr=GC_swr_ratID",num2str(rat),"_cbd;");
    eval(sprintf('%s',temp_var));


     temp_var = strcat( "GC_complex_swr=GC_complex_swr_ratID",num2str(rat),"_cbd;");
    eval(sprintf('%s',temp_var));


    %% Ripple
    wave_ripple_hpcbelo=[];
    for i=1:size(GC_ripple.waveforms,1)
        x=GC_ripple.waveforms{i,1}(3,:);
        wave_ripple_hpcbelo=[wave_ripple_hpcbelo; x];
    end
    save wave_ripple_hpcbelo.mat wave_ripple_hpcbelo
    FileData = load('wave_ripple_hpcbelo.mat');
    csvwrite('wave_ripple_hpcbelo.csv', FileData.wave_ripple_hpcbelo);

    clear FileData
    duration_ripple=[];
    for i=1:size(GC_ripple.grouped_oscil_table,1)
        x=table2array(GC_ripple.grouped_oscil_table(i,6:7));
        duration_ripple=[duration_ripple; x];
    end
    save duration_ripple.mat duration_ripple
    FileData = load('duration_ripple.mat');
    csvwrite('duration_ripple.csv', FileData.duration_ripple);

    %% SW
    wave_sw_hpcbelo=[];
    for i=1:size(GC_sw.waveforms,1)
        x=GC_sw.waveforms{i,1}(3,:);
        wave_sw_hpcbelo=[wave_sw_hpcbelo; x];
    end
    save wave_sw_hpcbelo.mat wave_sw_hpcbelo
    FileData = load('wave_sw_hpcbelo.mat');
    csvwrite('wave_sw_hpcbelo.csv', FileData.wave_sw_hpcbelo);

    clear FileData
    duration_sw=[];
    for i=1:size(GC_sw.grouped_oscil_table,1)
        x=table2array(GC_sw.grouped_oscil_table(i,6:7));
        duration_sw=[duration_sw; x];
    end
    save duration_sw.mat duration_sw
    FileData = load('duration_sw.mat');
    csvwrite('duration_sw.csv', FileData.duration_sw);

    %% SWR
    wave_swr_hpcbelo=[];
    for i=1:size(GC_swr.waveforms,1)
        x=GC_swr.waveforms{i,1}(3,:);
        wave_swr_hpcbelo=[wave_swr_hpcbelo; x];
    end
    save wave_swr_hpcbelo.mat wave_swr_hpcbelo
    FileData = load('wave_swr_hpcbelo.mat');
    csvwrite('wave_swr_hpcbelo.csv', FileData.wave_swr_hpcbelo);

    clear FileData
    duration_swr=[];
    for i=1:size(GC_swr.grouped_oscil_table,1)
        x=table2array(GC_swr.grouped_oscil_table(i,6:7));
        duration_swr=[duration_swr; x];
    end
    save duration_swr.mat duration_swr
    FileData = load('duration_swr.mat');
    csvwrite('duration_swr.csv', FileData.duration_swr);

    %% Complex SWR
    wave_complex_swr_hpcbelo=[];
    for i=1:size(GC_complex_swr.waveforms,1)
        x=GC_complex_swr.waveforms{i,1}(3,:);
        wave_complex_swr_hpcbelo=[wave_complex_swr_hpcbelo; x];
    end
    save wave_complex_swr_hpcbelo.mat wave_complex_swr_hpcbelo
    FileData = load('wave_complex_swr_hpcbelo.mat');
    csvwrite('wave_complex_swr_hpcbelo.csv', FileData.wave_complex_swr_hpcbelo);

    clear FileData
    duration_complex_swr=[];
    for i=1:size(GC_complex_swr.grouped_oscil_table,1)
        x=table2array(GC_complex_swr.grouped_oscil_table(i,6:7));
        duration_complex_swr=[duration_complex_swr; x];
    end
    save duration_complex_swr.mat duration_complex_swr
    FileData = load('duration_complex_swr.mat');
    csvwrite('duration_complex_swr.csv', FileData.duration_complex_swr);

    % PYTHON
    system('python compute-insta-freq.py');

    % features from .csv to .mat and replace
    features_ripple = readmatrix("features_ripple.csv");
    features_ripple=features_ripple(2:end,2);
    GC_ripple.HPCbelo_features(:,1)=features_ripple;

    features_sw = readmatrix("features_sw.csv");
    features_sw=features_sw(2:end,2);
    GC_sw.HPCbelo_features(:,1)=features_sw;

    features_swr = readmatrix("features_swr.csv");
    features_swr=features_swr(2:end,2);
    GC_swr.HPCbelo_features(:,1)=features_swr;

    features_complex_swr = readmatrix("features_complex_swr.csv");
    features_complex_swr=features_complex_swr(2:end,2);
    GC_complex_swr.HPCbelo_features(:,1)=features_complex_swr;
    
    
    %
    
     temp_var = strcat( "GC_ripple_ratID",num2str(rat),"_cbd=GC_ripple;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_sw_ratID",num2str(rat),"_cbd=GC_sw;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_swr_ratID",num2str(rat),"_cbd=GC_swr;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_complex_swr_ratID",num2str(rat),"_cbd=GC_complex_swr;");
    eval(sprintf('%s',temp_var));

    
    %
     temp_var = strcat( "save FTx_ratID",num2str(rat),"_cbd.mat data_info GC_ripple_ratID",num2str(rat),"_cbd GC_sw_ratID",num2str(rat),...
         "_cbd GC_swr_ratID",num2str(rat),"_cbd GC_complex_swr_ratID",num2str(rat),"_cbd");
    eval(sprintf('%s',temp_var));

    clearvars -except rat rats_cbd rats
    %delete 
end



