function [distance, path] = Dijkstra(a, sb, db)
    % a 邻接矩阵, sb起点标号, db终点标号, distance最短路距离，path最短路路径
    n = size(a, 1);
    L = zeros(n); L(1:n) = inf; L(sb) = 0;
    S = zeros(n); S(sb) = 1;
    Sh = 1 - S;
    lIndex = zeros(n); 
    path = [];
    
    for i = 1:n 
        lmin = inf; index = -1;
        for j = 1:n %update L(j)
            if Sh(j) == 0
                continue;
            else
                for k = 1:n
                    if S(k) == 1 
                        l = L(k) + a(j,k);
                        if l < L(j)
                            L(j) = l;
                            lIndex(j) = k;
                        end
                    else
                        continue;
                    end
                end
                if L(j) < lmin
                    lmin = L(j);
                    index = j;
                end
            end
        end
        S(index) = 1;
        Sh(index) = -1;
    end
    
    distance = L(db);
    
    p = db; path = [p, path];
    lp = lIndex(db);
    while lp ~= 0
        p = lp;
        lp = lIndex(p);
        path = [p, path];
    end
    
    
    
    