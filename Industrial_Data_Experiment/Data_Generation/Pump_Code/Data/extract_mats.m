load extracted_data_features.mat
x_all = data_arr(:,1:end-1);
y_all = data_arr(:,end);

client_number = 1;

mask = y_all == client_number;

x = x_all(mask,:);
y = y_all(mask);