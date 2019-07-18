clear all
close all

% create a sparse network (Erdos–Rényi graph)
n = 100; % number of nodes (agents)
weight = 0.1; % scaling of the edges weight (coupling strength)
% create adjacency matrix Ad
Ad = rand(n,n)*weight; % Adjacency matrix of the network
n_sparse = 0.7;  % probability that there is an edge between two nodes:1-n_sparse
zero_el = randperm(numel(Ad),n_sparse*numel(Ad));
Ad(zero_el) = 0;
%Ad=(Ad+Ad')/2; % for an undirected network
L = Ad-diag(sum(Ad')); % Laplacian matrix
% compute Ajacency and Laplacian matrices of the unweighted graph
Ad_unweight = ones(n,n); 
Ad_unweight(zero_el) = 0;
L_unweight = Ad_unweight-diag(sum(Ad_unweight'));

% Laplacian eigenvalues (what we try to recover from data)
lambda_L = eig(L);

% local linear dynamics (attached to each node); x_dot = A x + B u ; y = C x
A = [-1 -2;1 -1];
B = [1;2];
C = [1 1];
m = size(A,1); % number of local states

% Dynamics of the fulkl network: x_dot = Atot x
Atot = kron(eye(n),A)+kron(L,B*C);

% nonindentical neurons, choose sigma>0
sigma = 0; % standard deviation of heterogenity between the agents of the network
% compute the perturbation deltaA of Atot
deltaA = zeros(m*n,m*n);
for k = 1 : n
    deltaA((k-1)*m+1:k*m,(k-1)*m+1:k*m) = sigma*randn(m,m);
end

Atot_ident = Atot; % dynamics of the network with identical agents
Atot = Atot + deltaA; % dynamics of the network with nonidentical agents

%% measurements

% simu parameters
t_end = 20; % data obtained on the time interval [0,t_end]
nb_step = 50; % number of samples for each time series
pas = t_end/nb_step; % sampling time
nb_simus = 10; % number of time series
init_cond = randn(n*m,nb_simus); % initial condition for each time series

% compute the time series (solutions of x_dot = Atot x)
x = zeros(n*m*nb_simus,nb_step);
t = 0 : pas : t_end;
for j = 1 : length(t)
    x(:,j) = reshape(expm(t(j)*Atot)*init_cond,[n*m*nb_simus 1]);
end

% partial observations in the network
n_vertex = [1]; % measured nodes
n_state = [1]; % measured local states
nb_obs = 20; % total number of measures ; if nb_obs>nb_simus, then the same time series will be used several times (shifted with some delay delta_step)
delta_step = 5; % delay between two observations (along the same trajectory)

% extract the data (used for identification) from the time series
n_cut = ceil(nb_obs/nb_simus/length(n_vertex)/length(n_state));
f = [ ];
for i = n_vertex
    for j = n_state
        obs_states = n*m*([1:nb_simus]-1)+(i-1)*m+j;
        for k = 1 : n_cut
           f = [f;x(obs_states,1+(k-1)*delta_step:nb_step-(n_cut-(k-1))*delta_step)];
        end
    end
end
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
h1 = plot(real(lambda_fin),imag(lambda_fin),'xr','MarkerSize',18,'Linewidth',3);
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
h2 = plot(-real(guess_lambda),imag(guess_lambda),'xr','MarkerSize',18,'Linewidth',3);
xlabel('$\Re\{\lambda\}$','interpreter','latex','FontSize',24)
ylabel('$\Im\{\lambda\}$','interpreter','latex','rotation',90,'FontSize',24)
legend([h1,h2],'exact','measured')
