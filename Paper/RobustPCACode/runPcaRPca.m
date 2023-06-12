datasets = {'DSA', 'Isolet', 'MagicTelescope', 'Musk'};
rnks = [1 3 5 10 15 20 25 30 40 50 60 75 100 125 150];
alphas = [1 2 3 4 5];

results = struct([]);

nBytesDS = 0;

for di = 1:length(datasets)
    dstr = datasets{di};
    
    fprintf(repmat('\b', 1, nBytesDS));
    nBytesDS = fprintf('Dataset: %s\n', dstr);
    
    results(di).name = dstr;
    
    load([dstr, '/data.mat']);
    dim = size(trFeats, 2);
    
    rnkTst = rnks(rnks <= dim);
    tstAlps = (nnz(trLabs) / length(trLabs)) * alphas;
    tstAlps = tstAlps(tstAlps < 1);
    
    results(di).dim = dim;
    results(di).rnkTst = rnkTst;
    results(di).tstAlps = tstAlps;
    
    results(di).pcaAucs = [];
    results(di).pcaNacc = [];
    results(di).pcaPar = [];
    
    results(di).rPcaAucs = [];
    results(di).rPcaNacc = [];
    results(di).rPcaPar = [];
    
    
    maxRnk = max(rnkTst);
    
    nBytesRnk = 0;
    
    for rnk = rnkTst
        fprintf(repmat('\b', 1, nBytesRnk));
        nBytesRnk  = fprintf('Rank: %d\n', rnk);
        
        [~, ~, V] = lansvd(trFeats, rnk, 'L');
        scrs = getProjScrs(V, tstFeats);
        [auc, Par, Nacc] = compAucPNac(scrs, tstLabs);
        
        results(di).pcaAucs = [results(di).pcaAucs; auc];
        results(di).pcaNacc = [results(di).pcaNacc; Nacc];
        results(di).pcaPar = [results(di).pcaPar; Par];
        
        rPcaAuc = 0;
        rPcaNacc = 0;
        rPcaPar = 0;
        
        for alpha = tstAlps
            [V, ~, ~] = crpca(transpose(trFeats), rnk, 10, alpha);
            
            scrs = getProjScrs(V, tstFeats);
            
            [auc, Par, Nacc] = compAucPNac(scrs, tstLabs);
            
            rPcaAuc = max(auc, rPcaAuc);
            rPcaNacc = max(auc, rPcaNacc);
            rPcaPar = max(auc, rPcaPar);
        end
        
        results(di).rPcaAucs = [results(di).rPcaAucs; rPcaAuc];
        results(di).rPcaNacc = [results(di).rPcaNacc; rPcaNacc];
        results(di).rPcaPar = [results(di).rPcaPar; rPcaPar];
    end
    fprintf(repmat('\b', 1, nBytesRnk));
end
fprintf(repmat('\b', 1, nBytesDS));

load('ARHS/data.mat');

for sub = 1:2
    
    di = length(datasets) + sub;
    results(di).name = ['ARHS', num2str(sub)];
    
    eval(['trFeats = sub', num2str(sub), 'TrFeats']);
    eval(['tstFeats = sub', num2str(sub), 'TstFeats']);
    eval(['trLabs = sub', num2str(sub), 'TrLabs']);
    eval(['tstLabs = sub', num2str(sub), 'TstLabs']);
    
    dim = size(trFeats, 2);
    
    rnkTst = rnks(rnks <= dim);
    tstAlps = (nnz(trLabs) / length(trLabs)) * alphas;
    tstAlps = tstAlps(tstAlps < 1);
    
    results(di).dim = dim;
    results(di).rnkTst = rnkTst;
    results(di).tstAlps = tstAlps;
    
    results(di).pcaAucs = [];
    results(di).pcaNacc = [];
    results(di).pcaPar = [];
    
    results(di).rPcaAucs = [];
    results(di).rPcaNacc = [];
    results(di).rPcaPar = [];
    
    
    maxRnk = max(rnkTst);
    
    for rnk = rnkTst
        fprintf('Rank: %d\n', rnk);
        [~, ~, V] = lansvd(trFeats, rnk, 'L');
        scrs = getProjScrs(V, tstFeats);
        [auc, Par, Nacc] = compAucPNac(scrs, tstLabs);
        
        results(di).pcaAucs = [results(di).pcaAucs; auc];
        results(di).pcaNacc = [results(di).pcaNacc; Nacc];
        results(di).pcaPar = [results(di).pcaPar; Par];
        
        rPcaAuc = 0;
        rPcaNacc = 0;
        rPcaPar = 0;
        
        for alpha = tstAlps
            [V, ~, ~] = crpca(transpose(trFeats), rnk, 10, alpha);
            
            scrs = getProjScrs(V, tstFeats);
            
            [auc, Par, Nacc] = compAucPNac(scrs, tstLabs);
            
            rPcaAuc = max(auc, rPcaAuc);
            rPcaNacc = max(auc, rPcaNacc);
            rPcaPar = max(auc, rPcaPar);
        end
        
        results(di).rPcaAucs = [results(di).rPcaAucs; rPcaAuc];
        results(di).rPcaNacc = [results(di).rPcaNacc; rPcaNacc];
        results(di).rPcaPar = [results(di).rPcaPar; rPcaPar];
    end
end