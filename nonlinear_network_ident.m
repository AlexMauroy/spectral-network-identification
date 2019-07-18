clear all
close all

% create a sparse network (Erdos�R�nyi graph)
n = 10; % number of nodes (agents)
weight = 0.5; % scaling of the edges weight (coupling strength)
% create adjacency matrix Ad
Ad = rand(n,n)*weight; % Adjacency matrix of the network
n_sparse = 0.3;  % probability that there is an edge between two nodes:1-n_sparse
zero_el = randperm(numel(Ad),n_sparse*numel(Ad));
Ad(zero_el) = 0;
%Ad=(Ad+Ad')/2; % if we want an undirected network
L = Ad-diag(sum(Ad')); % Laplacian matrix
% compute Ajacency and Laplacian matrices of the unweighted graph
Ad_unweight = ones(n,n); 
Ad_unweight(zero_el) = 0;
L_unweight = Ad_unweight-diag(sum(Ad_unweight'));

% Laplacian eigenvalues (what we try to recover from data)
lambda_L = eig(L);

% nonlinear dynamics (attached to each node)
% parameters of linearized dynamics
A = [-1 -2;1 -1];
B = [1;1.5];
C = [1 1];

m = size(A,1); % number of local states

% nonindentical neurons, choose sigma>0
sigma = 0; % standard deviation of heterogenity between the agents
% compute the perturbation deltaA
deltaA = sigma*randn(n*m,1);

param.L = L;
param.deltaA = deltaA;

f_dyn     =   @(t,X) nonlinear_network_model(t,X,param);
options = odeset('RelTol',1e-9,'AbsTol',1e-300);

Atot_ident = kron(eye(n),A)+kron(L,B*C); % dynamics of the network with identical agents
Atot = Atot_ident+diag(deltaA); % dynamics of the network with nonidentical agents

%% measurements

% simu parameters
t_end = 40; % data obtained on the time interval [0,t_end]
nb_step = 80; % number of samples for each time series
pas = t_end/nb_step; % sampling time
nb_simus = 10; % number of time series
init_cond = -0.5+1*rand(n*m,nb_simus); % initial condition for each time series

% compute the time series
x = zeros(n*m*nb_simus,nb_step+1);
for j = 1 : nb_simus
    [t,y] = ode45(f_dyn,linspace(0,t_end,nb_step+1),init_cond(:,j),options);
    x((j-1)*n*m+1:j*n*m,:) = y';
end

% partial observations in the network
n_vertex = [1]; % measured nodes
n_state = [1 2]; % measured local states
nb_obs = 40; % total number of measures ; if nb_obs>nb_simus, then the same time series will be used several times (shifted with some delay delta_step)
delta_step = 5; % delay between two observations (along the same trajectory)

% extract the data (used for identification) from the time series
n_cut = ceil(nb_obs/nb_simus/length(n_vertex)/length(n_state));
f = [ ];
%f=x;
for i = n_vertex
    for j = n_state
        obs_states = n*m*([1:nb_simus]-1)+(i-1)*m+j;
        for k = 1 : n_cut
            f = [f;x(obs_states,1+(k-1)*delta_step:nb_step-(n_cut-(k-1))*delta_step)];
        end
    end
end

% % nonlinear observation function [sin(x_11+x_12) sin(x_11-x12)] ; see Example 2 in the submitted paper
% f = [f(1:nb_obs/2,:)+f(nb_obs/2+1:end,:);f(1:nb_obs/2,:)-f(nb_obs/2+1:end,:)];
% f = sin(f);

f_X = f(:,1:end-1);
f_Y = f(:,2:end);

%% DMD algorithm

nb_comput = 1; % number of runs of the DMD algorithm
n_traj = 1; % fraction of times series used for each run
n_snap = 1; % fraction of snapshots used for each run

lambda_fin = [];


for comput = 1 : nb_comput
    
    set1 = randperm(size(f_X,1)); % pick a random set of times series (when n_traj<1)
    set2 = randperm(size(f_X,2)); % pick a random set of snapshots (when n_snap<1)
    K = f_X(set1(1:floor(n_traj*size(f_X,1))),set2(1:floor(n_snap*size(f_X,2))));
    K2 = f_Y(set1(1:floor(n_traj*size(f_X,1))),set2(1:floor(n_snap*size(f_X,2))));

    % run DMD algo
    [eig_lambda V] = dmd_algo(K,K2);
    lambda_fin = [lambda_fin;log(eig_lambda)/pas]; % eigenvalues of Atot (estimated) 

end

% plot eigenvalues of Atot
figure(1)
hold on
lambda_Atot = eig(Atot);
lambda_Atot_ident = eig(Atot_ident);
h2 = plot(real(lambda_Atot),imag(lambda_Atot),'ob','MarkerSize',10,'Linewidth',3);
%h3 = plot(real(lambda_Atot_ident),imag(lambda_Atot_ident),'og','MarkerSize',3,'Linewidth',3);
%h1 = plot(real(lambda_fin),imag(lambda_fin),'xr','MarkerSize',18,'Linewidth',3);
h1 = plot(real(lambda_fin(1:n)),imag(lambda_fin(1:n)),'xr','MarkerSize',18,'Linewidth',3);
xlabel('$\Re\{\mu_k\}$','interpreter','latex')
ylabel('$\Im\{\mu_k\}$','interpreter','latex','rotation',90)
legend([h2,h1],'exact','measured')

%% reconstruction of Laplacian eigenvalues

clear guess_lambda

for k = 1 : length(lambda_fin)
   
    guess_lambda(k) = -1/(C*inv(A-lambda_fin(k)*eye(m))*B);

end

% plot Laplacian eigenvalues
figure(2)
h1 = plot(-real(lambda_L),imag(lambda_L),'ob','MarkerSize',10,'Linewidth',3);
hold on
box on
%h2 = plot(-real(guess_lambda),imag(guess_lambda),'xr','MarkerSize',18,'Linewidth',3);
h2 = plot(-real(guess_lambda(1:n)),imag(guess_lambda(1:n)),'xm','MarkerSize',18,'Linewidth',3);
xlabel('$\Re\{\lambda\}$','interpreter','latex','FontSize',24)
ylabel('$\Im\{\lambda\}$','interpreter','latex','rotation',90,'FontSize',24)
legend([h1,h2],'exact','measured')

% %% computation of spectral moments (for large networks)
% 
% % remove eigenvalues with large imaginary part
% lambda_fin(find(abs(imag(lambda_fin))>1.4)) = [];
% 
% % remove eigenvalues with large real part
% lambda_fin(find(abs(real(lambda_fin))>2)) = [];
% 
% % rescale the Laplacian eigenvalues (if necessary)
% %delta_x = max(abs(real(lambda_fin3)));
% %delta_y = max(abs(imag(lambda_fin3)));
% %scale = delta_x/delta_y;
% scale = 1;
% lambda_scale = real(lambda_fin)+1i*imag(lambda_fin)*scale;
% 
% % compute first spectral moment (M1)
% 
% moment_1A = 0; % initialization of the first spectral moment of Atot
% moment_2A = 0; % initialization of the second spectral moment of Atot
% % find clusters of eigenvalues
% nb_clusters = 2; % number of clusters
% [clust CM] = kmeans([real(lambda_scale) imag(lambda_scale)],nb_clusters);
% n_std = [1 1]; % distance used to remove outliers (the unit is the standard deviation of the cluster)
% for k = 1 : nb_clusters
%     
%     index = find(clust==k);
%     
%     % remove outliners
%     std = sqrt(sum(abs(lambda_scale(index)-CM(k,1)-1i*CM(k,2)).^2)/length(lambda_fin(index))); % standard deviation of the cluster
%     lambda_fin_hull = lambda_fin(index(find(abs(lambda_scale(index)-CM(k,1)-1i*CM(k,2))<n_std(k)*std))); % eigenvalues used to compute the convex hull of the cluster
%     
%     Khull = convhull(real(lambda_fin_hull),imag(lambda_fin_hull)); % convex hull of the cluster
%     
%     % plot the convex hull
%     figure(1)
%     patch(real(lambda_fin_hull(Khull)),imag(lambda_fin_hull(Khull)),[0.8 0.8 0.8],'linestyle','none')
%     
%     % compute the geometric moments of the convex hull
%     [GEOM_A{k}, INER_A{k}, CPMO_A{k} ] = polygeom(real(lambda_fin_hull(Khull)),imag(lambda_fin_hull(Khull)));
%     
%     moment_1A = moment_1A+GEOM_A{k}(2); % contribution to the first spectral moment of Atot
%     moment_2A = moment_2A+(INER_A{k}(2)-INER_A{k}(1))/GEOM_A{k}(1); % contribution to the second spectral moment of Atot
%     
% end
% 
% moment_1A = moment_1A/nb_clusters;
% moment_2A = moment_2A/nb_clusters;
% 
% M1_computed = (-m*moment_1A+trace(A))/(C*B) % first spectral moment of the Laplacian matrix
% 
% % M1 could be computed by summing all the Laplacian matrix (but it is less accurate, especially in th ecase of nonidentical agents)
% % M1_computed_bis = -sum(guess_lambda)/length(guess_lambda); 
% 
% M1 = -trace(L)/n % exact first spectral moment
% 
% % compute second spectral moment (M2)
% 
% M2_computed = (m*(moment_2A-sigma^2)-trace(A^2)+2*M1_computed*trace(A*B*C))/(C*B)^2 % second spectral moment of the Laplacian matrix
% M2 = trace(L^2)/n  % exact second spectral moment 
% 
% 
% %% other quantities
% 
% % mean node degree
% 
% av_weight = weight/2; % average weight
% var_weight = weight^2/12; % variance of weight distribution
% 
% mean_node_deg = M1_computed/av_weight % mean node degree
% exact_mean_node_deg = -trace(L_unweight)/n % exact value
% 
% M2_unweight = (M2_computed-var_weight*mean_node_deg)/av_weight^2 % second spectral moment of unweighted Laplacian matrix
% M2_unweight_exact = trace(L_unweight^2)/n % exact value