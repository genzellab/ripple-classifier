%% 

%% Generate data input for ripple classifier
clc
clear
addpath('/home/genzellab/Desktop/Pelin/someFigureJob/data');
addpath(genpath('/home/genzellab/Desktop/Pelin/someFigureJob/ADRITOOLS'));

%% Vehicle
clc
clear variables
rats_veh=[3 4 9 201 203 206 210 211 213];

for rat_no=1:size(rats_veh,2)
    clearvars -except -regexp rat
    clearvars -regexp GC

    rat=rats_veh(rat_no);

    %% x
    temp_var = strcat( "load('GC_ratID",num2str(rat),"_veh.mat')");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "x=GC_ripple_ratID",num2str(rat),"_veh;");
    eval(sprintf('%s',temp_var));

    Wave={};
    Wave2={};
    
    for i=1:length(x.waveforms)
        trace=x.waveforms{i}(2,:); %Extracting Pyr trace
        start_end=table2array([x.grouped_oscil_table(i,6:7)]);
        wave=trace(start_end(1):start_end(2));
        Wave=[Wave; {wave}];
    
        trace2=x.waveforms{i}(3,:);
        start_end=table2array([x.grouped_oscil_table(i,6:7)]);
        wave2=trace2(start_end(1):start_end(2));
        Wave2=[Wave2; {wave2}];
    end
    Wave=Wave.';
    Wave2=Wave2.';
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave,0,0,600,100,300);
    PCA_features=PCA_features(:,2:end);
    
    x.HPCpyra_trace=Wave';
    x.HPCpyra_features=PCA_features;
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave2,0,0,600,0.1,20);
    PCA_features=PCA_features(:,2:end);
    
    x.HPCbelo_trace=Wave2';
    x.HPCbelo_features=PCA_features;
    
        
    %% y
    temp_var = strcat( "y=GC_swr_ratID",num2str(rat),"_veh;");
    eval(sprintf('%s',temp_var));

    Wave={};
    Wave2={};
    for i=1:length(y.waveforms)
        i
        trace=y.waveforms{i}(2,:);
        start_end=table2array([y.grouped_oscil_table(i,6:7)]);
        wave=trace(start_end(1):start_end(2));
        Wave=[Wave; {wave}];
    
        trace2=y.waveforms{i}(3,:);
        start_end=table2array([y.grouped_oscil_table(i,6:7)]);
        wave2=trace2(start_end(1):start_end(2));
        Wave2=[Wave2; {wave2}];  
    end
    Wave=Wave.';
    Wave2=Wave2.';
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave,0,0,600,100,300);
    PCA_features=PCA_features(:,2:end);
    
    y.HPCpyra_trace=Wave';
    y.HPCpyra_features=PCA_features;
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave2,0,0,600,0.1,20);
    PCA_features=PCA_features(:,2:end);
    
    y.HPCbelo_trace=Wave2';
    y.HPCbelo_features=PCA_features;
    
    
    %% z
    temp_var = strcat( "z=GC_complex_swr_ratID",num2str(rat),"_veh;");
    eval(sprintf('%s',temp_var));

    Wave={};
    Wave2={};
    for i=1:length(z.waveforms)
        i
        trace=z.waveforms{i}(2,:);
        start_end=table2array([z.grouped_oscil_table(i,6:7)]);
        wave=trace(start_end(1):start_end(2));
        Wave=[Wave; {wave}];
    
        trace2=z.waveforms{i}(3,:);
        start_end=table2array([z.grouped_oscil_table(i,6:7)]);
        wave2=trace2(start_end(1):start_end(2));
        Wave2=[Wave2; {wave2}]; 
    end
    Wave=Wave.';
    Wave2=Wave2.';
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave,0,0,600,100,300);
    PCA_features=PCA_features(:,2:end);
    
    z.HPCpyra_trace=Wave';
    z.HPCpyra_features=PCA_features;
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave2,0,0,600,0.1,20);
    PCA_features=PCA_features(:,2:end);
    
    z.HPCbelo_trace=Wave2';
    z.HPCbelo_features=PCA_features;
    
    %% t (SW)
    temp_var = strcat( "t=GC_sw_ratID",num2str(rat),"_veh;");
    eval(sprintf('%s',temp_var));

    Wave={};
    Wave2={};
    for i=1:length(t.waveforms)
        i
        trace=t.waveforms{i}(2,:);
        start_end=table2array([t.grouped_oscil_table(i,6:7)]);
        wave=trace(start_end(1):start_end(2));
        Wave=[Wave; {wave}];
    
        trace2=t.waveforms{i}(3,:);
        start_end=table2array([t.grouped_oscil_table(i,6:7)]);
        wave2=trace2(start_end(1):start_end(2));
        Wave2=[Wave2; {wave2}];
    end
    Wave=Wave.';
    Wave2=Wave2.';
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave,0,0,600,100,300);
    PCA_features=PCA_features(:,2:end);
    
    t.HPCpyra_trace=Wave';
    t.HPCpyra_features=PCA_features;
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave2,0,0,600,0.1,20);
    PCA_features=PCA_features(:,2:end);
    
    t.HPCbelo_trace=Wave2';
    t.HPCbelo_features=PCA_features;
   

    %% Now overwrite GC_ratID2_veh.mat with x
    temp_var = strcat( "GC_ripple_ratID",num2str(rat),"_veh=x;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_swr_ratID",num2str(rat),"_veh=y;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_complex_swr_ratID",num2str(rat),"_veh=z;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_sw_ratID",num2str(rat),"_veh=t;");
    eval(sprintf('%s',temp_var));
    
    %% remove previous fields
    clearvars -except -regexp GC data_info rat

    %
    temp_var = strcat( "GC_ripple_ratID",num2str(rat),"_veh = rmfield(GC_ripple_ratID",num2str(rat),"_veh,'ripple_trace');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_ripple_ratID",num2str(rat),"_veh = rmfield(GC_ripple_ratID",num2str(rat),"_veh,'PCA_features');");
    eval(sprintf('%s',temp_var));
    %
    temp_var = strcat( "GC_swr_ratID",num2str(rat),"_veh = rmfield(GC_swr_ratID",num2str(rat),"_veh,'ripple_trace');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_swr_ratID",num2str(rat),"_veh = rmfield(GC_swr_ratID",num2str(rat),"_veh,'PCA_features');");
    eval(sprintf('%s',temp_var));
    %
    temp_var = strcat( "GC_complex_swr_ratID",num2str(rat),"_veh = rmfield(GC_complex_swr_ratID",num2str(rat),"_veh,'ripple_trace');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_complex_swr_ratID",num2str(rat),"_veh = rmfield(GC_complex_swr_ratID",num2str(rat),"_veh,'PCA_features');");
    eval(sprintf('%s',temp_var));


    %  
    clearvars -except -regexp GC data_info rat
    clearvars rat_no rats_veh
    temp_var = strcat( "save GC_ratID",num2str(rat),"_veh.mat");
    eval(sprintf('%s',temp_var));
    rats_veh=[3 4 9 201 203 206 210 211 213];
end


%% CBD
clc
clear variables
rats_cbd=[2 5 10 11 204 205 207 209 212 214];

for rat_no=1:size(rats_cbd,2)
    clearvars -except -regexp rat
    clearvars -regexp GC

    rat=rats_cbd(rat_no);

    %% x
    temp_var = strcat( "load('GC_ratID",num2str(rat),"_cbd.mat')");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "x=GC_ripple_ratID",num2str(rat),"_cbd;");
    eval(sprintf('%s',temp_var));

    Wave={};
    Wave2={};
    
    for i=1:length(x.waveforms)
        trace=x.waveforms{i}(2,:); %Extracting Pyr trace
        start_end=table2array([x.grouped_oscil_table(i,6:7)]);
        wave=trace(start_end(1):start_end(2));
        Wave=[Wave; {wave}];
    
        trace2=x.waveforms{i}(3,:);
        start_end=table2array([x.grouped_oscil_table(i,6:7)]);
        wave2=trace2(start_end(1):start_end(2));
        Wave2=[Wave2; {wave2}];
    end
    Wave=Wave.';
    Wave2=Wave2.';
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave,0,0,600,100,300);
    PCA_features=PCA_features(:,2:end);
    
    x.HPCpyra_trace=Wave';
    x.HPCpyra_features=PCA_features;
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave2,0,0,600,0.1,20);
    PCA_features=PCA_features(:,2:end);
    
    x.HPCbelo_trace=Wave2';
    x.HPCbelo_features=PCA_features;
    
        
    %% y
    temp_var = strcat( "y=GC_swr_ratID",num2str(rat),"_cbd;");
    eval(sprintf('%s',temp_var));

    Wave={};
    Wave2={};
    for i=1:length(y.waveforms)
        i
        trace=y.waveforms{i}(2,:);
        start_end=table2array([y.grouped_oscil_table(i,6:7)]);
        wave=trace(start_end(1):start_end(2));
        Wave=[Wave; {wave}];
    
        trace2=y.waveforms{i}(3,:);
        start_end=table2array([y.grouped_oscil_table(i,6:7)]);
        wave2=trace2(start_end(1):start_end(2));
        Wave2=[Wave2; {wave2}];  
    end
    Wave=Wave.';
    Wave2=Wave2.';
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave,0,0,600,100,300);
    PCA_features=PCA_features(:,2:end);
    
    y.HPCpyra_trace=Wave';
    y.HPCpyra_features=PCA_features;
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave2,0,0,600,0.1,20);
    PCA_features=PCA_features(:,2:end);
    
    y.HPCbelo_trace=Wave2';
    y.HPCbelo_features=PCA_features;
    
    
    %% z
    temp_var = strcat( "z=GC_complex_swr_ratID",num2str(rat),"_cbd;");
    eval(sprintf('%s',temp_var));

    Wave={};
    Wave2={};
    for i=1:length(z.waveforms)
        i
        trace=z.waveforms{i}(2,:);
        start_end=table2array([z.grouped_oscil_table(i,6:7)]);
        wave=trace(start_end(1):start_end(2));
        Wave=[Wave; {wave}];
    
        trace2=z.waveforms{i}(3,:);
        start_end=table2array([z.grouped_oscil_table(i,6:7)]);
        wave2=trace2(start_end(1):start_end(2));
        Wave2=[Wave2; {wave2}]; 
    end
    Wave=Wave.';
    Wave2=Wave2.';
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave,0,0,600,100,300);
    PCA_features=PCA_features(:,2:end);
    
    z.HPCpyra_trace=Wave';
    z.HPCpyra_features=PCA_features;
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave2,0,0,600,0.1,20);
    PCA_features=PCA_features(:,2:end);
    
    z.HPCbelo_trace=Wave2';
    z.HPCbelo_features=PCA_features;
    
    %% t (SW)
    temp_var = strcat( "t=GC_sw_ratID",num2str(rat),"_cbd;");
    eval(sprintf('%s',temp_var));

    Wave={};
    Wave2={};
    for i=1:length(t.waveforms)
        i
        trace=t.waveforms{i}(2,:);
        start_end=table2array([t.grouped_oscil_table(i,6:7)]);
        wave=trace(start_end(1):start_end(2));
        Wave=[Wave; {wave}];
    
        trace2=t.waveforms{i}(3,:);
        start_end=table2array([t.grouped_oscil_table(i,6:7)]);
        wave2=trace2(start_end(1):start_end(2));
        Wave2=[Wave2; {wave2}];
    end
    Wave=Wave.';
    Wave2=Wave2.';
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave,0,0,600,100,300);
    PCA_features=PCA_features(:,2:end);
    
    t.HPCpyra_trace=Wave';
    t.HPCpyra_features=PCA_features;
    
    [~,~,~,~,~,~,~,~,~,~,PCA_features]=delta_specs(Wave2,0,0,600,0.1,20);
    PCA_features=PCA_features(:,2:end);
    
    t.HPCbelo_trace=Wave2';
    t.HPCbelo_features=PCA_features;
   

    %% Now overwrite GC_ratID2_cbd.mat with x
    temp_var = strcat( "GC_ripple_ratID",num2str(rat),"_cbd=x;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_swr_ratID",num2str(rat),"_cbd=y;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_complex_swr_ratID",num2str(rat),"_cbd=z;");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_sw_ratID",num2str(rat),"_cbd=t;");
    eval(sprintf('%s',temp_var));
    
    %% remove previous fields
    clearvars -except -regexp GC data_info rat

    %
    temp_var = strcat( "GC_ripple_ratID",num2str(rat),"_cbd = rmfield(GC_ripple_ratID",num2str(rat),"_cbd,'ripple_trace');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_ripple_ratID",num2str(rat),"_cbd = rmfield(GC_ripple_ratID",num2str(rat),"_cbd,'PCA_features');");
    eval(sprintf('%s',temp_var));
    %
    temp_var = strcat( "GC_swr_ratID",num2str(rat),"_cbd = rmfield(GC_swr_ratID",num2str(rat),"_cbd,'ripple_trace');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_swr_ratID",num2str(rat),"_cbd = rmfield(GC_swr_ratID",num2str(rat),"_cbd,'PCA_features');");
    eval(sprintf('%s',temp_var));
    %
    temp_var = strcat( "GC_complex_swr_ratID",num2str(rat),"_cbd = rmfield(GC_complex_swr_ratID",num2str(rat),"_cbd,'ripple_trace');");
    eval(sprintf('%s',temp_var));

    temp_var = strcat( "GC_complex_swr_ratID",num2str(rat),"_cbd = rmfield(GC_complex_swr_ratID",num2str(rat),"_cbd,'PCA_features');");
    eval(sprintf('%s',temp_var));


    %  
    clearvars -except -regexp GC data_info rat
    clearvars rat_no rats_cbd
    temp_var = strcat( "save GC_ratID",num2str(rat),"_cbd.mat");
    eval(sprintf('%s',temp_var));
    rats_cbd=[2 5 10 11 204 205 207 209 212 214];
end





