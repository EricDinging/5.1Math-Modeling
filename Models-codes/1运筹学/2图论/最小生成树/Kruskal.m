function result = Kruskal(a)
    [i, j, b] = find(a);
    data = [i'; j'; b']; 
    index = data(1:2, :);
    loop = length(a) - 1;
    result = [];
    while length(result) < loop
        temp = min(data(3, :));
        flag = find(data(3, :) == temp);
        flag = flag(1);
        v1 = index(1, flag); v2 = index(2, flag); 
        if v1 ~= v2
            result = [result, data(:, flag)];
            index(index == v1) = v2;
        end
        data(:, flag) = []; index(:, flag) = [];
    end