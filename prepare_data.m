
%% Load data

% Specify DATETIME format
opts = detectImportOptions('oecd.csv');
opts = setvaropts(opts,'DATE','InputFormat','dd/MM/uuuu');

oecd = readtable("oecd.csv",opts);

%% Prepare data

% Get unique countries and dates
countries = unique(oecd.REF_AREA);
dates = sort(unique(oecd.DATE)).';

% Get number of unique countries and dates
no_countries = length(countries);
no_dates = length(dates);

% Pre-allocate tensor
X_oecd = zeros(no_countries,3,no_dates);

% Loop countries
for ind_cntry=1:no_countries
    % Loop dates
    for ind_dt=1:no_dates
        
        % Get country and date
        cntry = string(countries(ind_cntry));
        dt = dates(ind_dt);
        
        % Extract the data for the current country and date
        stir = oecd{oecd.REF_AREA == cntry & oecd.DATE == dt, "SHORT_TERM_INTEREST_RATE"};
        hicp = oecd{oecd.REF_AREA == cntry & oecd.DATE == dt, "HICP"};
        rtv = oecd{oecd.REF_AREA == cntry & oecd.DATE == dt, "RETAIL_TRADE_VOLUME"};
        
        % Assign data to tensor row fibers
        X_oecd(ind_cntry,1,ind_dt) = stir;
        X_oecd(ind_cntry,2,ind_dt) = hicp;
        X_oecd(ind_cntry,3,ind_dt) = rtv;
    end
end

% Subtract mean from time series
X_mean = mean(X_oecd, 3);
X_oecd = X_oecd - X_mean;

% Detrend time series
for ind_cntry=1:no_countries

    X_oecd(ind_cntry,1,:) = detrend(squeeze(X_oecd(ind_cntry,1,:)));
    X_oecd(ind_cntry,2,:) = detrend(squeeze(X_oecd(ind_cntry,2,:)));
    X_oecd(ind_cntry,3,:) = detrend(squeeze(X_oecd(ind_cntry,3,:)));
end

% Export prepared data
save("X_oecd","X_oecd")