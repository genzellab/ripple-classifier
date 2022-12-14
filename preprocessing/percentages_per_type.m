
n=[ 3 4 9 201 203 206 210 211 213];
%n=[ 2 5 10 11 204 205 207 209 212 214];

str1='HPCpyra_events_ratID';
%VEH
cd ('/home/adrian/Documents/SWR_classifier/wetransfer_per_rat/veh/HPCpyra')

for i=1:length(n)
str=[str1 num2str(n(i)) '.mat']
load(str);

complex=size(HPCpyra_complex_swr_veh,1);
ripple=size(HPCpyra_ripple_veh,1);
swr=size(HPCpyra_swr_veh,1);

total=swr+ripple+complex

p_ripples(i)=(ripple/total)*100;
p_complex(i)=(complex/total)*100;
p_swr(i)=(swr/total)*100;
end

n=[ 2 5 10 11 204 205 207 209 212 214];
%VEH
cd ('/home/adrian/Documents/SWR_classifier/wetransfer_per_rat/cbd/HPCpyra')

for i=1:length(n)
str=[str1 num2str(n(i)) '.mat']
load(str);

complex=size(HPCpyra_complex_swr_cbd,1);
ripple=size(HPCpyra_ripple_cbd,1);
swr=size(HPCpyra_swr_cbd,1);

total=swr+ripple+complex

p_ripples_cbd(i)=(ripple/total)*100;
p_complex_cbd(i)=(complex/total)*100;
p_swr_cbd(i)=(swr/total)*100;
end
% 
% bar(p_complex)
% bar(p_ripples)
% bar(p_swr)


boxplot([p_ripples.', p_swr.', p_complex.'])
%%
SEM_ripples = std(p_ripples.')./ sqrt(length(p_ripples));                                % Calculate Standard Error Of The Mean
SEM_ripples_cbd = std(p_ripples_cbd.')./ sqrt(length(p_ripples_cbd));                                % Calculate Standard Error Of The Mean


% SEM_ripples=[mean(p_ripples)-SEM_ripples mean(p_ripples)+SEM_ripples]

%%
SEM_swr = std(p_swr.')./ sqrt(length(p_swr));                                % Calculate Standard Error Of The Mean
SEM_swr_cbd = std(p_swr_cbd.')./ sqrt(length(p_swr_cbd));                                % Calculate Standard Error Of The Mean

% SEM_swr=[mean(p_swr)-SEM_swr mean(p_swr)+SEM_swr];
    
%%

SEM_complex =[ std(p_complex.')./ sqrt(length(p_complex))];                                % Calculate Standard Error Of The Mean
SEM_complex_cbd =[ std(p_complex_cbd.')./ sqrt(length(p_complex_cbd))];                                % Calculate Standard Error Of The Mean

% SEM_complex=[mean(p_complex)-SEM_complex mean(p_complex)+SEM_complex]
%% VEH only
%x = 1:3
x = categorical({'Ripples', 'SWR','cSWR'}); %xaxis
x = reordercats(x,{'Ripples', 'SWR','cSWR'}); %specified order
y=[mean(p_ripples) mean(p_swr) mean(p_complex)];
bar(x,y)
hold on
e = errorbar(x,y,[SEM_ripples SEM_swr SEM_complex],'o');    
e.Marker = '*';
e.MarkerSize = 10;
e.Color = 'black';
e.LineWidth=2;

ylabel('Percentage of total events')

title('Percentage of event type across veh rats')
% xlim([0 4])
ylim([0 70])
%% VEH and CBD
x = categorical({'Ripples', 'SWR','cSWR'}); %xaxis
x = reordercats(x,{'Ripples', 'SWR','cSWR'}); %specified order
model_error=[SEM_ripples SEM_ripples_cbd; SEM_swr SEM_swr_cbd; SEM_complex SEM_complex_cbd];

% x = categorical({'Ripples','Ripples CBD', 'SWR','SWR CBD','cSWR','cSWR CBD'}); %xaxis
% x = reordercats(x,{'Ripples','Ripples CBD', 'SWR','SWR CBD','cSWR','cSWR CBD'}); %specified order
y=[mean(p_ripples) mean(p_ripples_cbd); mean(p_swr) mean(p_swr_cbd); mean(p_complex) mean(p_complex_cbd)];
% b=bar(x,reshape(y,2,3).')
b=bar(y,'grouped')
hold on
%%
% Calculate the number of groups and number of bars in each group
[ngroups,nbars] = size(y);
% Get the x coordinate of the bars
X = nan(nbars, ngroups);
for i = 1:nbars
    X(i,:) = b(i).XEndPoints;
end

%%
e = errorbar(X.',y,model_error,'k','linestyle','none');    
%e.Marker = '*';
%e.MarkerSize = 10;
% e.Color = 'black';
% e.LineWidth=2;

ylabel('Percentage of total events')

title('Percentage of event type across veh rats')
% xlim([0 4])
ylim([0 70])

%%
plotColors = jet(length(p_ripples));
hold on
for i=1:length(p_ripples)
%       Plot_color=cmap(i./5,:);   
plot(X(1,:).',[ p_ripples(i),p_swr(i),p_complex(i)],'o','MarkerFaceColor','k')
%     ax = gca;
%     ax.ColorOrderIndex = i;
end

for i=1:length(p_ripples_cbd)
%       Plot_color=cmap(i./5,:);   
plot(X(2,:).',[ p_ripples_cbd(i),p_swr_cbd(i),p_complex_cbd(i)],'o','MarkerFaceColor','k')
%     ax = gca;
%     ax.ColorOrderIndex = i;
end

%%

%%

plotColors = jet(length(p_ripples));
hold on
for i=1:length(p_ripples)
%       Plot_color=cmap(i./5,:);   
plot(x,[ p_ripples(i),p_swr(i),p_complex(i)],'-o','MarkerFaceColor','k','LineWidth',2,'Color', plotColors(i,:))
%     ax = gca;
%     ax.ColorOrderIndex = i;
end

%legend({'a','Mean+SEM','Rat3','Rat4','Rat9','Rat201','Rat203','Rat206','Rat210','Rat211','Rat213'})