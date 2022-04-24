% 加入用户界面：三个按钮（运行，停止，退出），一个文本框（仿真运行次数）
plotbutton = uicontrol('Style','pushbutton','String','Run',...
    'FontSize',12,'Position',[100,400,50,20],'Callback','run=1');
erasebutton = uicontrol("Style",'pushbutton','String','Stop',...
    'FontSize',12,'Position',[200,400,50,20],'Callback','freeze=1');
quitbutton = uicontrol('Style','pushbutton','String','Quit',...
    'FontSize',12,'Position',[300,400,50,20],'Callback','stop=1;close');
number = uicontrol('Style','text','String','1',...
    'FontSize',12,'Position',[20,400,50,20]);

% 初始化：元胞状态为0，中心十字形的元胞状态为1
n = 200;
z = zeros(n,n);
cells = z;
cells(n/2, .25*n:.75*n) = 1;
cells(.25*n:.75*n, n/2) = 1;
sum = z;
stop = 0;
run = 0;
freeze = 0;

% 更新元胞范围
x = 2:n-1;
y = 2:n-1;

% 建立RGB图像，返回句柄
        imh = image(cat(3,cells,z,z));
        set(imh,'erasemode','none');
        axis equal
        axis tight

% 循环
while(stop == 0)
    if(run == 1)
        % 邻居规则
        sum(x,y) = cells(x,y-1) + cells(x,y+1) + ...
            cells(x-1,y) + cells(x+1,y) + ...
            cells(x-1,y-1) + cells(x-1,y+1) + ...
            cells(x+1,y-1) + cells(x+1,y+1);
        % CA规则
        cells = (sum == 3) | (sum == 2 & cells);
        % 画图
        set(imh, 'cdata', cat(3,cells,z,z));
        % 更新模拟次数
        stepnumber = 1 + str2double(get(number,'string'));
        set(number,'string',num2str(stepnumber));
    end
    if(freeze == 1)
        run = 0;
        freeze = 0;
    end
drawnow
end

