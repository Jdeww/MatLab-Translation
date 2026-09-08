% 1). Dual window detection 
%     Refr: "Multiple-Window Anomaly Dectection for Hyperspectral Imagery"
%     by W.M. Liu, C.I. Chang
% 2). RXD with variable size window

close all; clear all;
tic;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% with autoenconder
%f = dlmread('hymap-deepenc4_400.csv'); M=100; N=300; L=113; D=f;   % 11/7455; 13/7279 15/6938
load ('F_random_anom3.mat'); M=86; N=86; D=f; L=100;
load('gt_random_anom3.mat'); [MN L] = size(D);%

MN = M*N;
mx_sz = [M N];
X = zeros(M,N,L);
for i=1:L
  X(:,:,i) = reshape(D(:,i),mx_sz);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% HyMap
OuterWindowSz = 11; InnerWindowSz = 1;  % 6734

OuterRadius = (OuterWindowSz-1)/2;
InnerRadius = (InnerWindowSz-1)/2;
COutWin = OuterWindowSz * OuterWindowSz;
CInWin = InnerWindowSz * InnerWindowSz;
DiffRadius = OuterWindowSz - InnerWindowSz;
detRXD = zeros(M,N);
one = ones(L,1);

CM_outer = []; CM_inner = [];
for i=OuterRadius+1: M-(OuterRadius)
  if (mod(i,10) == 0) 
      disp(['Please wait ... i=' num2str(i) ' out of ' num2str(M)]);
  end    
  for j=OuterRadius+1: N-(OuterRadius)
    CM_outer = []; CM_inner = [];

    % Find anomaly
    for l=1:L;
      M_outer=X(i-OuterRadius:i+OuterRadius,j-OuterRadius:j+OuterRadius,l);
      M_inner=X(i-InnerRadius:i+InnerRadius,j-InnerRadius:j+InnerRadius,l);

      CM_outer = [CM_outer reshape(M_outer,COutWin,1)]; 
      CM_inner = [CM_inner reshape(M_inner,CInWin,1)];
    end
    
    if (InnerWindowSz > 1)
      K_outer = cov(CM_outer); 
      K_inner = cov(CM_inner);
      u_outer = mean(CM_outer);
      u_inner = mean(CM_inner);
      K_diff = K_outer - K_inner;
      u_diff = u_outer - u_inner;
      u_diff = u_diff';

      % DWEST
      [evec eval] = eig(K_diff);
      v = zeros(L,1);
      z = [L-5 L-4 L-3 L-2 L-1 L];  % use greatest 6 e-vectors
      for k=z(1):z(length(z))
        v = v + evec(:,k);
      end
      detRXD(i,j) = v' * u_diff;
    else
      r = squeeze(X(i,j,:));
      r = r + rand(size(r))*0.7;
      K_outer = cov(CM_outer); 
      K_inv = inv(K_outer);
      u =mean(CM_outer(:));
      U = repmat(u,L,1);
      detRXD(i,j) = (r-U)' * K_inv * (r-U);
      %%%detRXD(i) = (one-U)' * K_inv * (r-U);  % UTD = 0
    end
  end
end

disp(['Please wait ... i=' num2str(M) ' out of ' num2str(M)]);

figure; 
imshow(abs(detRXD),[]);
axis image; axis off;
title(['Outer window sz=' num2str(OuterWindowSz) ...
  ', Inner window sz=' num2str(InnerWindowSz)]);

detRXD = reshape(detRXD,1,MN);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Produce ROC

x = detRXD;
x_max = max(x(:)); x_min = min(x(:)); x_denom = x_max-x_min;
x = (x-repmat(x_min,1,MN))/x_denom * 1000;
x = reshape(x,[M N]);

for i=0:999
  b = x > i;
  nd(i+1) = nnz(b & gt);
  nf(i+1) = nnz(b & ~gt);
  thr(i+1) = i;
end

pd = nd/nnz(gt);
pf = nf/nnz(~gt);
figure; plot(pf,pd);
axis([0 1 0 1]); grid;
xlabel('False Alarms'); ylabel('Detected');
title ('ROC');

toc;
area = 1+trapz(pd,pf)

