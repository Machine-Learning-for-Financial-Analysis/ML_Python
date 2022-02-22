tic;

addpath("/MATLAB Drive/Private")

temp = readmatrix(fullfile(matlabdrive, 'Private', 'OxfordManRealizedVolatility-2000-2016-original variable_nonames.csv'));  % change this line and the one below to where you have saved the spreadsheets of data

%dates = temp(:,1);
data = temp(:,4:end).*10000;  % RV	RQ	RJ	BPV	RV_min	RV_plus	TPQ	MedRQ	TrRQ	RQ15min	 RQboot

T = size(data,1);
Explanatory_variables=[];

for i=1:size(data,2)
    clear tempVar tempVard tempVarw tempVarm;
    tempVar=data(:,i);
    tempVard = tempVar;
    tempVarw = mean([tempVar,mlag(tempVar,4,mean(tempVar))],2);
    tempVarm = mean([tempVar,mlag(tempVar,21,mean(tempVar))],2);
    Explanatory_variables=[Explanatory_variables tempVard(22:end-1) tempVarw(22:end-1) tempVarm(22:end-1)];
end

RV  = data(:,1);
RVd = RV;

writematrix(Explanatory_variables,'Explanatory_variables.csv')

Response_variable=RVd(23:end);

writematrix(Response_variable,'Response_variable.csv')
