
%% Imports

addpath('../../Tools/Matlab/tensorlab_2016-03-28/')
addpath('../../0_git/functional-tucker-decomposition/statistics/')
addpath('../../0_git/functional-tucker-decomposition/statistics/kernels/')
addpath('../../0_git/functional-tucker-decomposition/algorithm/')

%% Build models (naive, spline, ftd)

%% First the optimal parameter d is determined

% Define time stamps and subset for training
xs = 1:264;
xs_tr = 1:4:264;

% Select training data
X_tr = X_oecd(:,:,xs_tr);

% Define training parameters
LAMBDA = 1;
MAXITERS = 25;
TAU = 1e-9;
ds = linspace(1,7.5,21);

% Pre-allocate container for storing RMSEs
all_rmse_1_ftd = zeros(size(ds,2),1);
all_rmse_2_ftd = zeros(size(ds,2),1);
all_rmse_3_ftd = zeros(size(ds,2),1);

% Pre-allocate container for storing approximation error
all_approxerror_ftd = zeros(size(ds,2),1);

% Loop training parameter d
for d_ind=1:size(ds,2)
    
    d = ds(d_ind);
    disp("Run d="+d)

    % Build FTD model
    % Get kernel matrices for training and inference
    K_tr_tr = kernel_gaussian(xs_tr,xs_tr,d); 
    K_te_tr = kernel_gaussian(xs,xs_tr,d); 
    
    % Compute FTD
    [Us_ftd_class,S_ftd_class,W_ftd_class,~] = f_hooi_gradient_exp_clean(X_tr,K_tr_tr,[9 3 21],LAMBDA,MAXITERS,TAU);

    % Save approximation error
    all_approxerror_ftd(d_ind,1) = frob(X_tr-tmprod(S_ftd_class,Us_ftd_class,1:ndims(X_tr)))/frob(X_tr);
    
    % Get new factor matrix
    C_ftd_xs_te_class = K_te_tr*W_ftd_class;
    
    % Compute reconstruction with new factor matrix
    X_ftd_te_class_reconstr = tmprod(S_ftd_class, {Us_ftd_class{1:2},C_ftd_xs_te_class}, 1:3);

    % Compute RMSEs
    rmse_1_ftd = rmse(squeeze(X_ftd_te_class_reconstr(:,1,:)),squeeze(X_oecd(:,1,xs)),"all");
    rmse_2_ftd = rmse(squeeze(X_ftd_te_class_reconstr(:,2,:)),squeeze(X_oecd(:,2,xs)),"all");
    rmse_3_ftd = rmse(squeeze(X_ftd_te_class_reconstr(:,3,:)),squeeze(X_oecd(:,3,xs)),"all");
        
    % Save RMSEs
    all_rmse_1_ftd(d_ind,1) = rmse_1_ftd;
    all_rmse_2_ftd(d_ind,1) = rmse_2_ftd;
    all_rmse_3_ftd(d_ind,1) = rmse_3_ftd;
end

%% Visualization: RMSEs and approximation errors of FTD

% Setup
% Colorblind-safe (Okabe–Ito) palette
C1 = [0    114 178]/255;   % blue
C2 = [213  94  0  ]/255;   % vermillion
C3 = [0    158 115]/255;   % bluish green
C4 = [230  159 0]/255;      % orange

% Common style
lw  = 2.2;                  % line width
ms  = 6.5;                  % marker size
fs = 18;                    % font size

f = figure;
tl=tiledlayout(1,2);

% Relative approximation error
nexttile
plot(ds, all_approxerror_ftd, 'o-', 'Color',C2, ...
    'LineWidth',lw,'Marker','*','MarkerFaceColor',C2); 

% Axes 
set(gca,'FontName','Times','FontSize',12,'Color','none', ...    
        'LineWidth',2, 'TickDir','out', ...
        'Box','off', 'XMinorTick','on','YMinorTick','on');   
axis tight
xlabel('d'); % Label for the x-axis
xlim([min(ds) max(ds)])
ylabel('Rel. approx. error'); 

% Legend
legend("Approx. error of FTD (training data)", ...
        'Location','north','Orientation','vertical', ...
        'Interpreter','none','FontSize',12,'Color','none');

% RMSEs
nexttile
plot(ds, all_rmse_1_ftd ./ no_countries, '*-', 'Color',C1, ...
    'LineWidth',lw,'Marker','*','MarkerFaceColor',C1); hold on;
plot(ds, all_rmse_2_ftd ./ no_countries, '*-', 'Color',C2, ...
    'LineWidth',lw,'Marker','*','MarkerFaceColor',C2); hold on;
plot(ds, all_rmse_3_ftd ./ no_countries, '*-', 'Color',C3, ...
    'LineWidth',lw,'Marker','*','MarkerFaceColor',C3); hold off;

% Axes 
set(gca,'FontName','Times','FontSize',12,'Color','none', ...    
        'LineWidth',2, 'TickDir','out', ...
        'Box','off', 'XMinorTick','on','YMinorTick','on');   
axis tight
xlabel('d'); % Label for the x-axis
xlim([min(ds) max(ds)])
ylabel('RMSE'); 

% Legend
legend(["SHORT_TERM_INTEREST_RATE","HICP","RETAIL_TRADE_VOLUME"], ...
        'Location','north','Orientation','vertical', ...
        'Interpreter','none','FontSize',12,'Color','none');

% The illustration suggests an optimal d=4

%% Build the FTD model with optimal d and compare it to naive and splines

% Select country to consider
i = 1; % Example for Belgium

% Select parameter d
d = 4;

% Build FTD model
% Get kernel matrices for training and inference
K_tr_tr = kernel_gaussian(xs_tr,xs_tr,d); 
K_te_tr = kernel_gaussian(xs,xs_tr,d); 

% Compute FTD
[Us_ftd_class,S_ftd_class,W_ftd_class,errs]  = f_hooi_gradient_exp_clean(X_tr,K_tr_tr,[9 3 21],LAMBDA,MAXITERS,TAU);

% Get new factor matrix
C_ftd_xs_te_class = K_te_tr*W_ftd_class;

% Compute reconstruction with new factor matrix
X_ftd_te_class_reconstr = tmprod(S_ftd_class, {Us_ftd_class{1:2},C_ftd_xs_te_class}, 1:3);

% Build naive models
X_i_1_naive = repelem(squeeze(X_tr(i,1,:)),4);
X_i_2_naive = repelem(squeeze(X_tr(i,2,:)),4);
X_i_3_naive = repelem(squeeze(X_tr(i,3,:)),4);

% Build spline models
X_i_1_spline = interp1(xs_tr,squeeze(X_tr(i,1,:)),xs,'spline');
X_i_2_spline = interp1(xs_tr,squeeze(X_tr(i,2,:)),xs,'spline');
X_i_3_spline = interp1(xs_tr,squeeze(X_tr(i,3,:)),xs,'spline');

% Get FTD models
X_i_1_ftd = squeeze(X_ftd_te_class_reconstr(i,1,:));
X_i_2_ftd = squeeze(X_ftd_te_class_reconstr(i,2,:));
X_i_3_ftd = squeeze(X_ftd_te_class_reconstr(i,3,:));
    
% Compute rmses (for x_start to x_last)
x_first = 205;
x_last = 264;
x_first_last = x_first:x_last;
xs_first_last = xs(x_first:x_last);
rmse_i_1_original = rmse(squeeze(X_oecd(i,1,x_first_last)),squeeze(X_oecd(i,1,x_first_last)));
rmse_i_1_naive = rmse(X_i_1_naive(x_first_last),squeeze(X_oecd(i,1,x_first_last)));
rmse_i_1_spline = rmse(X_i_1_spline(x_first_last).',squeeze(X_oecd(i,1,x_first_last)));
rmse_i_1_ftd = rmse(squeeze(X_ftd_te_class_reconstr(i,1,x_first_last)),squeeze(X_oecd(i,1,x_first_last)));
rmse_i_2_original = rmse(squeeze(X_oecd(i,2,x_first_last)),squeeze(X_oecd(i,2,x_first_last)));
rmse_i_2_naive = rmse(X_i_2_naive(x_first_last),squeeze(X_oecd(i,2,x_first_last)));
rmse_i_2_spline = rmse(X_i_2_spline(x_first_last).',squeeze(X_oecd(i,2,x_first_last)));
rmse_i_2_ftd = rmse(squeeze(X_ftd_te_class_reconstr(i,2,x_first_last)),squeeze(X_oecd(i,2,x_first_last)));
rmse_i_3_original = rmse(squeeze(X_oecd(i,3,x_first_last)),squeeze(X_oecd(i,3,x_first_last)));
rmse_i_3_naive = rmse(X_i_3_naive(x_first_last),squeeze(X_oecd(i,3,x_first_last)));
rmse_i_3_spline = rmse(X_i_3_spline(x_first_last).',squeeze(X_oecd(i,3,x_first_last)));
rmse_i_3_ftd = rmse(squeeze(X_ftd_te_class_reconstr(i,3,x_first_last)),squeeze(X_oecd(i,3,x_first_last)));

%% Visualization: Interpolation results

% Plot SHORT TERM INTEREST RATE
f1 = figure;
plot(xs_first_last, X_i_1_naive(x_first:x_last),'-','Color',C2,'LineWidth',lw); hold on;
plot(xs_first_last, X_i_1_spline(x_first:x_last),'-','Color',C3,'LineWidth',lw); hold on;
plot(xs_first_last, X_i_1_ftd(x_first:x_last),'-','Color',C4,'LineWidth',lw); hold on;
plot(xs_first_last, squeeze(X_oecd(i,1,x_first:x_last)),'Marker','diamond','MarkerSize',6,'MarkerFaceColor',C1,'MarkerEdgeColor',C1,'LineStyle','none'); hold off;

% Axes 
set(gca,'FontName','Times','FontSize',fs,'Color','none','LineWidth',1.2, 'TickDir','out', 'Box','off', 'XMinorTick','on','YMinorTick','on'); 
axis tight

% Set x-tick labels to display months
xticks(xs(x_first+3:5:x_last));  % Force ticks at every data point
xticklabels(string(dates(x_first+3:5:x_last)));  % Apply all labels

% Legend
legend(["Naive (rmse=" + rmse_i_1_naive + ")", ...
        "Spline (rmse=" + rmse_i_1_spline + ")", ...
        "FTD (rmse=" + rmse_i_1_ftd + ")", ...
        "Original (rmse=" + rmse_i_1_original + ")"], ...
        'Location','southeast','Orientation','vertical','Interpreter','none','FontSize',fs,'Color','none');

title("SHORT TERM INTEREST RATE");

% Plot HICP
f2 = figure;
plot(xs_first_last, X_i_2_naive(x_first:x_last),'-','Color',C2,'LineWidth',lw); hold on;
plot(xs_first_last, X_i_2_spline(x_first:x_last),'-','Color',C3,'LineWidth',lw); hold on;
plot(xs_first_last, X_i_2_ftd(x_first:x_last),'-','Color',C4,'LineWidth',lw); hold on;
plot(xs_first_last, squeeze(X_oecd(i,2,x_first:x_last)),'Marker','diamond','MarkerSize',6,'MarkerFaceColor',C1,'MarkerEdgeColor',C1,'LineStyle','none'); hold off;

% Axes 
set(gca,'FontName','Times','FontSize',fs,'Color','none','LineWidth',1.2, 'TickDir','out', 'Box','off', 'XMinorTick','on','YMinorTick','on'); 
axis tight

% Set x-tick labels to display months
xticks(xs(x_first+3:5:x_last));  % Force ticks at every data point
xticklabels(string(dates(x_first+3:5:x_last)));  % Apply all labels

% Legend
legend(["Naive (rmse=" + rmse_i_2_naive + ")", ...
        "Spline (rmse=" + rmse_i_2_spline + ")", ...
        "FTD (rmse=" + rmse_i_2_ftd + ")", ...
        "Original (rmse=" + rmse_i_2_original + ")"], ...
        'Location','southeast','Orientation','vertical','Interpreter','none','FontSize',fs,'Color','none');

title("HICP");

% Plot RETAIL TRADE VOLUME

f3 = figure;
plot(xs_first_last, X_i_3_naive(x_first:x_last),'-','Color',C2,'LineWidth',lw); hold on;
plot(xs_first_last, X_i_3_spline(x_first:x_last),'-','Color',C3,'LineWidth',lw); hold on;
plot(xs_first_last, X_i_3_ftd(x_first:x_last),'-','Color',C4,'LineWidth',lw); hold on;
plot(xs(x_first:x_last), squeeze(X_oecd(i,3,x_first:x_last)),'Marker','diamond','MarkerSize',6,'MarkerFaceColor',C1,'MarkerEdgeColor',C1,'LineStyle','none'); hold off;

% Axes 
set(gca,'FontName','Times','FontSize',fs,'Color','none','LineWidth',1.2, 'TickDir','out', 'Box','off', 'XMinorTick','on','YMinorTick','on'); 
axis tight

% Set x-tick labels to display months
xticks(xs(x_first+3:5:x_last));  % Force ticks at every data point
xticklabels(string(dates(x_first+3:5:x_last)));  % Apply all labels

% Legend
legend(["Naive (rmse=" + rmse_i_3_naive + ")", ...
        "Spline (rmse=" + rmse_i_3_spline + ")", ...
        "FTD (rmse=" + rmse_i_3_ftd + ")", ...
        "Original (rmse=" + rmse_i_3_original + ")"], ...
        'Location','southeast','Orientation','vertical','Interpreter','none','FontSize',fs,'Color','none');

title("RETAIL TRADE VOLUME");

