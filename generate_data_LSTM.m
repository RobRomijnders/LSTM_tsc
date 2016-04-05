% Generate dummy data for LSTM

clear all
close all
clc


data_train = load('Two_Patterns_TEST');


X_train = data_train(:,2:end);
y_train = data_train(:,1);
clearvars data_train

N = size(X_train,1);
D = size(X_train,2);

%Generate four sinusoids
t = linspace(0,1,D);
fs = [2,3,4,5];
xs = ones(length(fs),D);
figure
hold on
for i = 1:length(fs)
    xs(i,:) = cos(2*pi*t*fs(i));
    plot(xs(i,:))
end
hold off

X_train_ex = zeros(N,D);
y_train_ex = zeros(N,1);

for i = 1:N
    class = randi([1 4],1,1);
    X_train_ex(i,:) = xs(class,:);
    y_train_ex(i) = class;
end

data = [y_train_ex X_train_ex];

%%
csvwrite('data_train_dummy',data)

csvwrite('data_test_dummy',data)
