
n=[ 3 4 9 201 203 206 210 211 213];
str1='HPCpyra_events_ratID';

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



bar(p_complex)
bar(p_ripples)
bar(p_swr)


boxplot([p_ripples.', p_swr.', p_complex.'])
%%
SEM_ripples = std(p_ripples.')./ sqrt(length(p_ripples));                                % Calculate Standard Error Of The Mean
% SEM_ripples=[mean(p_ripples)-SEM_ripples mean(p_ripples)+SEM_ripples]

%%
SEM_swr = std(p_swr.')./ sqrt(length(p_swr));                                % Calculate Standard Error Of The Mean
% SEM_swr=[mean(p_swr)-SEM_swr mean(p_swr)+SEM_swr];
    
%%

SEM_complex =[ std(p_complex.')./ sqrt(length(p_complex))];                                % Calculate Standard Error Of The Mean
% SEM_complex=[mean(p_complex)-SEM_complex mean(p_complex)+SEM_complex]
%%
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

title('Percentage of event type across Veh rats')
% xlim([0 4])
ylim([0 70])
%%

plotColors = jet(length(p_ripples));
hold on
for i=1:length(p_ripples)
%       Plot_color=cmap(i./5,:);   
plot(x,[ p_ripples(i),p_swr(i),p_complex(i)],'-o','MarkerFaceColor','k','LineWidth',2,'Color', plotColors(i,:))
%     ax = gca;
%     ax.ColorOrderIndex = i;
end

legend({'a','Mean+SEM','Rat3','Rat4','Rat9','Rat201','Rat203','Rat206','Rat210','Rat211','Rat213'})