%% Plot with enhanced land visibility
figure('Position', [100 100 1200 500]);

% Subplot 1: Sea Ice Concentration
subplot(1,2,1);
ax1 = axesm('MapProjection', 'stereo', 'MapLatLimit', [40 90], ...
    'Origin', [90 0], 'Frame', 'on', 'Grid', 'on', ...
    'MLineLocation', 30, 'PLineLocation', 10);
pcolorm(lat_A, lon_avhrr_360, reshape(SIT_monthly(40,4,:,:),361,361));
shading interp;
hold on;

% White land with black borders for better contrast
geoshow('landareas.shp', 'FaceColor', [1 1 1], 'EdgeColor', 'k', 'LineWidth', 1);

colorbar;
% caxis([0 100]);
title('Sea Ice Concentration (Interpolated)', 'FontSize', 12, 'FontWeight', 'bold');
colormap(gca, 'jet');

% Subplot 2: Sea Ice Thickness
subplot(1,2,2);
ax2 = axesm('MapProjection', 'stereo', 'MapLatLimit', [40 90], ...
    'Origin', [90 0], 'Frame', 'on', 'Grid', 'on', ...
    'MLineLocation', 30, 'PLineLocation', 10);
pcolorm(lat_A, lon_avhrr_360, test);
shading interp;
hold on;

% White land with black borders
geoshow('landareas.shp', 'FaceColor', [1 1 1], 'EdgeColor', 'k', 'LineWidth', 1);

colorbar;
title('Sea Ice Thickness (SIT April)', 'FontSize', 12, 'FontWeight', 'bold');
colormap(gca, 'parula');
