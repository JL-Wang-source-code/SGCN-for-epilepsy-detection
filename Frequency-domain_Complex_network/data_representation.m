function [dataset] = graph_representation(data, method)
[n,m,~] = size(data);
complex_network = [];
self = eye(m);
if(method == 'H')
for i = 1:n
    complex_network = LPhorizontal1(data(i,:),0);
    complex_network = complex_network + self;
    dataset(:,:,i) = complex_network;
end
elseif(method == 'LH')
for i = 1:n
    complex_network = LPhorizontal1(data(i,:),2);
    complex_network = complex_network + self;
    dataset(:,:,i) = complex_network;
end 
elseif(method == 'V')
for i = 1:n
    complex_network = LPvisibility1(data(i,:),0);
    complex_network = complex_network + self;
    dataset(:,:,i) = complex_network;
end
elseif(method == 'LV')
for i = 1:n
    complex_network = LPvisibility1(data(i,:),2);
    complex_network = complex_network + self;
    dataset(:,:,i) = complex_network;
end
else
    fprintf('No such metohd!');
end
for k = 1:n
    dataset(:,:,k) = dataset(:,:,k)/max(max(dataset(:,:,k)));
end  
train_data = [dataset(:,:,[(1:1200),(1601:2800)])];
test_data = [dataset(:,:,[(1201:1600),(2801:3200)])];
train_label = [zeros(1200,1);ones(1200,1)];
test_label = [zeros(400,1);ones(400,1)];
dataset_name = inputname(1);
save(['F:\dataset\',dataset_name,'_', method,'.mat'],'train_data','train_label','test_data','test_label')
