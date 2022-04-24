clc, clear;
% 加入用户界面：三个按钮（运行，停止，退出），一个文本框（仿真运行次数）
plotbutton = uicontrol('Style','pushbutton','String','Run',...
    'FontSize',12,'Position',[100,400,50,20],'Callback','run=1');
erasebutton = uicontrol("Style",'pushbutton','String','Stop',...
    'FontSize',12,'Position',[200,400,50,20],'Callback','freeze=1');
quitbutton = uicontrol('Style','pushbutton','String','Quit',...
    'FontSize',12,'Position',[300,400,50,20],'Callback','stop=1;close');
number = uicontrol('Style','text','String','1',...
    'FontSize',12,'Position',[20,400,50,20]);

% 设定坐标系是一个固定的尺寸，把坐标系里写入文本，然后获得并返回坐标内的内容
ax = axes('units','pixels','position',[1 1 500 400],'color','k');
text('units','pixels','position',[130,255,0],...
	'string','MCM','color','w','fontname','helvetica','fontsize',100);
text('units','pixels','position',[10,120,0],...
'string','Cellular Automata','color','w','fontname','helvetica','fontsize',50);
initial = getframe(gca); % 用getframe把他们写入一个矩阵

% 初始化
[a,b,c] = size(initial.cdata);
z = zeros(a,b);
cells = double(initial.cdata(:,:,1)==255);
visit = z;
sum = z;
threshold = 0.5;

% 初始化状态
stop = 0;
run = 0;
freeze = 0;

% 建立RGB图像，返回句柄
        imh = image(cat(3,cells,z,z));
        set(imh,'erasemode','none');
        axis equal
        axis tight

% 循环
while(stop == 0)
    if(run == 1)
        % 邻居规则
        sum(2:a-1,2:b-1) = cells(2:a-1,1:b-2) + cells(2:a-1,3:b) + ...
            cells(1:a-2,2:b-1) + cells(3:a,2:b-1) + ...
            cells(1:a-2,1:b-2) + cells(1:a-2,3:b) + ...
            cells(3:a,1:b-2) + cells(3:a,3:b);
        % CA规则
        pick = rand(a,b);
        cells = cells | ((sum>=1) & (pick>=threshold) & (visit==0));
        visit = (sum>=1);
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