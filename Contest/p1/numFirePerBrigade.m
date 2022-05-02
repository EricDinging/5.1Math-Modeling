clear, clc;
k = 27860;
num = zeros(1, 18);
case1 = "是 (True)"; case2 = "否，其中有1次为真实火灾 (False, one of them was a real fire)"; case3 = "否 (False)";
data = readtable('alarm.xlsx');
machineNum = table2array(data(1:k,1));
loop = table2array(data(1:k,2));
alarmNum = table2array(data(1:k,6));
obj = string(table2cell(data(1:k,9)));
brigade = string(table2cell(data(1:k,10)));
result = string(table2cell(data(1:k,11)));

index = find(result ~= case1);
machineNum = machineNum(index);loop = loop(index);alarmNum = alarmNum(index);obj = obj(index);brigade = brigade(index);result = result(index);

for i = 1:18
    brigadeName = convertCharsToStrings(join([char(i+64),'大队 (Fire brigade ',char(i+64),')']));
    brigadeInd = find(brigade == brigadeName);
    brigadeMachineNum = machineNum(brigadeInd); brigadeLoop = loop(brigadeInd);
    brigadeAlarmNum = alarmNum(brigadeInd); brigadeObj = obj(brigadeInd);
    brigadeResult = result(brigadeInd);
    num(i) = calculateNumFire(brigadeMachineNum, brigadeLoop, brigadeAlarmNum, brigadeObj, brigadeResult, case2);
end

dates = 18;
area = [1712, 692, 1100, 1631, 412, 1524, 122, 532, 96, 58, 1831, 1561, 1997, 246, 483, 24, 2151, 13];
perAreaFrequency = num / dates ./ area;
disp(perAreaFrequency);

function num = calculateNumFire(machineNum, loop, alarmNum, obj, result, case2)
    num = 0;
    for i = 1:length(machineNum)
        maxcases = 0;
        for j = 1:i - 1
            if ~isempty(j) && strcmp(obj(j), obj(i)) && machineNum(j) == machineNum(i) && loop(j) == loop(i)
                if result(j) == case2
                    temp = 1;
                else
                    temp = alarmNum(j);
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
end