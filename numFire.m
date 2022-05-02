clear,clc;
k = 27860;

data = readtable('alarm.xlsx');
machineNum = table2array(data(1:k,1));
loop = table2array(data(1:k,2));
alarmNum = table2array(data(1:k,6));
obj = string(table2cell(data(1:k,9)));
result = string(table2cell(data(1:k,11)));
case1 = "是 (True)"; case2 = "否，其中有1次为真实火灾 (False, one of them was a real fire)"; case3 = "否 (False)";
index = find(result ~= case1);
num = 0;

for i = index'
    maxcases = 0;
    for j = find(index < i)'
        if ~isempty(j) && strcmp(obj(index(j)), obj(i)) && machineNum(index(j)) == machineNum(i) && loop(index(j)) == loop(i)
            if result(index(j)) == case2
                temp = 1;
            else
                temp = alarmNum(index(j));
            end
            if temp > maxcases
                maxcases = temp;
            end      
        end
    end
    if result(i) == case2
        val = 1;
    else
        val = alarmNum(i);
    end
    if val > maxcases
        num = num - maxcases + val;
    end
end
disp(num);
