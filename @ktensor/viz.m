function info = viz(K, varargin)
%VIZ Visualize a ktensor.
%
%   VIZ(K) visualizes the components of an D-way ktensor with R components
%   in an R x D arrangment of plots. Each column of plots represents the
%   columns of the associated factor matrix, and each row represents a
%   single rank-one component.
%
%   INFO = VIZ(K, parameter, value...) takes optional parameters and
%   returns additional information from the plot, including handles to all
%   the axes.
%
%   Optional parameters:
%
%   -- Figure --
%   'Figure' - Figure number. Default: [] (new figure).
%
%   -- Plot Details --
%   'Plottype' - Cell array of plot types per mode. Default: {'line',...}.
%       o 'line' - Normal line plot. Default linewidth is 1.
%       o 'bar' - Bar plot.
%       o 'scatter' - Scatter plot. Default markersize is 10.
%   'Plotsize' - Array of sizes linewidth of markers. Default:-1*ones(D,1).
%   'Plotcolors' - Cell array of colors. Can be single color or array of
%                  colors for scatter or bar plots.
%   'Sameylims' - Use the same ylimits on all axes in the same mode.
%
%   -- Titles --
%   'Modetitles' - Cell array of mode titles. Default:{'Mode 1',...}
%   'Factortitles' - Factor title as number or weight. Default: 'None'.
%       o 'Weight' - Print relative lambda value, compared to lambda(1). 
%       o 'Number' - Print factor number.
%       o 'None' - No factor titles (default).
%
%   -- Spacing (all proportions in [0,1]) --
%   'Relmodespace' - Relative vertical space per mode. Default: ones(D,1).
%   'Vspacetop' - Space at the top, for titles. Default: 0.05.
%   'Vspacebottom' - Space at bottom, for xticklabels. Default: 0.05.
%   'Hspaceleft' - Space at left, for component labels. Default: 0.025.
%   'Hspaceright' - Space at right. Default: 0.025.
%   'Vspace' - Vertical space inbetween factor axes. Default: 0.01.
%   'Hspace' - Horizontal space inbetween factor axes. Default: 0.01.
%
%   Return values:
%   'height' - Height of each plot (as a proportion in [0,1]).
%   'width' - Width of each plot (as a proportion in [0,1]).
%   'ModeTitles' - Handles for the D mode titles.
%   'GlobalAxis' - Handle for main axes in which all others are embedded.
%   'FactorAxes' - R x D array of handles for the subfigure axes.
%   'htitle' - D-array of the handles to the mode titles (on the top).
%   'hftitle' - F-array of hanldes to the factor titles (on the left).
%   'h'- D x R array of handles to the figures for each factor.
%   
%   Examples:
%   K = ktensor([3; 2], rand(40,2), rand(50,2), rand(30,2));
%   viz(K,'Figure',1,'Hspace',0.05,'Vspacebottom',0.075);
%
%   Thanks to Alex Williams for the prototype for this functionality.
%
%MATLAB Tensor Toolbox.
%Copyright 2017, Sandia Corporation.

% TGK: Need to add options around line at zero, marks to use, font sizes, etc.



%%
nd = ndims(K); % Order
nc = ncomponents(K); % Rank

% parse optional inputs


params = inputParser;
% Figure 
params.addParameter('Figure', []);
% Spacing
params.addParameter('Relmodespace', ones(nd,1)); % Horizontal space for each mode
params.addParameter('Hspace',0.01); % Horizontal space between axes
params.addParameter('Hspaceright',0.025); % Horizontal space on left
params.addParameter('Hspaceleft',0.025); % Horizontal space on right
params.addParameter('Vspace',0.01); % Vertical space between axes
params.addParameter('Vspacetop',0.05); % Vertical space at top
params.addParameter('Vspacebottom',0.05); % Vertical space at bottom
% Titles
params.addParameter('Modetitles', []);
params.addParameter('Factortitle', 'none'); % Default is 'none'. Options are 'weight' or 'number'
% Plots
params.addParameter('Plottype', repmat({'line'}, [nd 1]));
params.addParameter('Plotsize', -1 * ones(nd,1)); % Used for scatter dot size or plot linewidth
params.addParameter('Plotcolors', cell(nd,1));
params.addParameter('Sameylims', true(nd,1));


params.parse(varargin{:});
res = params.Results;

%% Create new figure or reset old figure
if isempty(res.Figure)
    figure;
else
    figure(res.Figure);
    clf;
end

%% Create axes

% Calculate the amount of vertical space available for the plots themselves
% by subtracting off the top and bottom space as well as the inbetween
% space.
Vplotspace = 1 - res.Vspacetop - res.Vspacebottom - (nc - 1) * res.Vspace;
height = Vplotspace / nc;

% Do likewise for the horizontal space.
Hplotspace = 1 - res.Hspaceleft - res.Hspaceright - (nd - 1) * res.Hspace;
width = (res.Relmodespace ./ sum(res.Relmodespace)) .* Hplotspace;

% Create the global axis
GlobalAxis = axes('Position',[0 0 1 1]); % Global Axes
axis off;

% Create the nc x nd factor axes array
FactorAxes = gobjects(nc,nd); % Factor Axes
for k = 1 : nd
    for j = 1 : nc
        xpos = res.Hspaceleft + (k-1) * res.Hspace + sum(width(1:k-1));
        ypos = 1 - res.Vspacetop - height - (j-1) * (height + res.Vspace);
        FactorAxes(j,k) = axes('Position',[xpos ypos width(k) height]);
        FactorAxes(j,k).FontSize = 14;
    end
end


%% Plot each factor
h = gobjects(nd,nc);
for k = 1 : nd
    
    % Grab appropriate size
    if res.Plotsize(k) == -1
        lw = 1;
        ss = 10;
    else
        lw = res.Plotsize(k);
        ss = res.Plotsize(k);
    end

    % Grab appropriate colors
    if isempty(res.Plotcolors{k})
        cc = [0 0 1];
    else
        cc = res.Plotcolors{k};
    end
    
    % Extract component, no modifications
    U = K.u{k};
    
    % Add one extra at end of ticks
    xl = [0 size(K,k)+1];

    % Create y-axes that include zero
    yl = [min( 0, min(U(:)) ), max( 0, max(U(:)) )];

    for j = 1 : nc
        
        xx = 1:size(K,k);
        yy = U(:,j);
        hold(FactorAxes(j,k), 'off');

        switch res.Plottype{k}
            case 'line'
                hh = plot(FactorAxes(j,k), xx, yy, 'Linewidth', lw, 'Color', cc);
            case 'scatter'
                hh = scatter(FactorAxes(j,k), xx, yy, ss, cc, 'filled');              
            case 'bar'
                hh = bar(FactorAxes(j,k), xx, yy, 'EdgeColor', cc, 'FaceColor', cc);
        end
        
        % Set x-axes
        xlim(FactorAxes(j,k),xl);
        
        % Set y-axes
        if res.Sameylims(k)
            ylim(FactorAxes(j,k),yl);
        else
            % Create y-axes that include zero
            tmpyl = [ min(-0.01, min(U(:,j))), max( 0.01, max(U(:,j))) ];
            ylim(FactorAxes(j,k),tmpyl);
        end
        
        % Turn off y-ticks
        set(FactorAxes(j,k),'Ytick',[]);
        
        % Draw a box around the axes
        set(FactorAxes(j,k),'Box','on')

        % Turn of x-labels if not the bottom plot
        if j < nc
            set(FactorAxes(j,k),'XtickLabel',{});
        end            
        
        % Draw dashed line at zero
        hold(FactorAxes(j,k), 'on');
        plot(FactorAxes(j,k), xl, [0 0], 'k:', 'Linewidth', 1.5);

        % Save handle for main plot
        h(k,j) = hh;
        
        % Make the fonts on the xtick labels big
        set(FactorAxes(j,k),'FontSize',14)
    end
end

%% Title for each mode
htitle = gobjects(nd,1);
if ( isscalar(res.Modetitles) && islogical(res.Modetitles) && (res.Modetitles == false) )
    ModeTitles = 'none';
else
    if isempty(res.Modetitles)
        ModeTitles = cell(nd,1);
        for i = 1:nd
            ModeTitles{i} = sprintf('Mode %d',i);
        end
    else
        ModeTitles = res.Modetitles;
    end
    
    axes(GlobalAxis);
    for k = 1:nd
        xpos = res.Hspaceleft + (k-1) * res.Hspace + sum(width(1:k-1)) + 0.5 * width(k);
        %xpos = res.Hspaceleft + (k-1) * (width + res.Hspace) + 0.5 * width;
        ypos = 1 - res.Vspacetop;
        htitle(k) = text(xpos,ypos,ModeTitles{k},'VerticalAlignment','Bottom','HorizontalAlignment','Center');
        set(htitle(k),'FontSize',16)
        set(htitle(k),'FontWeight','bold')
    end
end

%% Print factor titles
hftitle = gobjects(nc,1);
if ~strcmpi(res.Factortitle,'none')
    axes(GlobalAxis);
    rellambda = abs (K.lambda / K.lambda(1));
    for j = 1:nc
        xpos = 0.9 * res.Hspaceleft;
        ypos = 1 - res.Vspacetop - 0.5 * height - (j-1) * (height + res.Vspace);
        %ypos = 1 - res.Vspacetop - 0.5 * height - (j-1) * (1 + res.Vrelspace) * height;
        if strcmpi(res.Factortitle,'weight')          
            txt = sprintf('%3.2f', rellambda(j));
        else
            txt = sprintf('%d', j);
        end
        hftitle(j) = text(xpos,ypos,txt,'VerticalAlignment','Middle','HorizontalAlignment','Right');
        set(hftitle(j),'FontSize',14)
    end
end
%% Save stuff to return
info.height = height;
info.width = width;
info.ModeTitles = ModeTitles;
info.GlobalAxis = GlobalAxis;
info.FactorAxes = FactorAxes;
info.htitle = htitle;
info.hftitle = hftitle;
info.h = h;

