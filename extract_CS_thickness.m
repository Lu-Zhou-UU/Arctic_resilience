% clear;
% YY = linspace(1982,2025,44);
% load('AVHRR_albedo_hi.mat');
% lat_A = double(lat);
% lon = double(lon);
% lon_360 = lon;
% lon_360(lon < 0) = lon(lon < 0) + 360;
% 
% list = dir('/Volumes/Yotta_1/C3C_seaice/ice_thickness_nh_ease2-250_*-v3p0_*10.nc');
% 
% for i = 1:numel(list)
%     hi = double(ncread(sprintf('%s/%s',list(i).folder,list(i).name),'sea_ice_thickness'));
%     lat = double(ncread(sprintf('%s/%s',list(i).folder,list(i).name),'lat'));
%     hi(find(lat<40 | lat>81.5)) = NaN;
%     mass(i) = nansum(hi(:).*25.*25./1000);
%     YYYY(i) = str2num(list(i).name(end-8:end-5));
% end
% figure;plot(YYYY,mass,'ko');
% % % 
% fp = fopen('C3_ice_volume_October.txt','wt');
% 
% % Write data
% for i = 1:numel(YYYY)
%     fprintf(fp, '%d %.3f\n', YYYY(i), mass(i));
% end
% fclose(fp);

clear;
YY = linspace(1979,2024,46);
list_Area = dir('/Volumes/Yotta_1/PIOMAS/area.*.nc');
list_Heff = dir('/Volumes/Yotta_1/PIOMAS/heff.*.nc');

k = 1;
for i = 1:numel(list_Area)
    hi = double(ncread(sprintf('%s/%s',list_Heff(i).folder,list_Heff(i).name),'heff'));
    sic = double(ncread(sprintf('%s/%s',list_Area(i).folder,list_Area(i).name),'area'));
    lat = double(ncread(sprintf('%s/%s',list_Heff(i).folder,list_Heff(i).name),'lat_scaler'));
    dxt = double(ncread(sprintf('%s/%s',list_Heff(i).folder,list_Heff(i).name),'dxt'));
    dyt = double(ncread(sprintf('%s/%s',list_Heff(i).folder,list_Heff(i).name),'dyt'));
    dxy = dxt.*dxt;
    hi(find(hi>100)) = NaN;
    sic(find(sic>100 | sic<0)) = NaN;
    for j = 1:12
        TT = hi(:,:,j).*sic(:,:,j).*dxy./1000;
        TT(find(lat<40 | lat>81.5)) = NaN;
        Mass_PIOAMS(k) = nansum(nansum(TT));       
        k = k+1;
    end
    % mass(i) = nansum(nansum(hi.*sic.*dxy./1000));
    % YYYY(i) = str2num(list(i).name(end-8:end-5));
end

fp = fopen('PIOMAS_ice_volume_October.txt','wt');
% Write data
for i = 1:numel(YY)
    fprintf(fp, '%d %.3f\n', YY(i), Mass_PIOAMS(10+(i-1)*12));
end
fclose(fp);

fp = fopen('PIOMAS_ice_volume_April.txt','wt');
% Write data
for i = 1:numel(YY)
    fprintf(fp, '%d %.3f\n', YY(i), Mass_PIOAMS(4+(i-1)*12));
end
fclose(fp);


% % % 
% % % % Close the file
% % % 
% % % % 
% % albedo = double(ncread("CERES_SYN1deg-Month_Terra-Aqua-NOAA20_Ed4.2_Subset_200003-202505.nc",'ini_albedo_mon'));
% % lat_ceres = double(ncread("CERES_SYN1deg-Month_Terra-Aqua-NOAA20_Ed4.2_Subset_200003-202505.nc",'lat'));
% % lon_ceres = double(ncread("CERES_SYN1deg-Month_Terra-Aqua-NOAA20_Ed4.2_Subset_200003-202505.nc",'lon'));
% % 
% list = dir('Month_SIC/ice_conc_nh_ease2-*.nc');
% lat_conc = double(ncread(sprintf('%s/%s',list(1).folder,list(1).name),'lat'));
% lon_conc = double(ncread(sprintf('%s/%s',list(1).folder,list(1).name),'lon'));
% % 
% % Convert ice concentration longitudes from [-180,180] to [0,360] to match CERES
% lon_conc_360 = lon_conc;
% lon_conc_360(lon_conc < 0) = lon_conc(lon_conc < 0) + 360;
% 
% % % Create meshgrids for interpolation
% % [LON_CERES, LAT_CERES] = meshgrid(lon_ceres, lat_ceres);
% % % Initialize array to store interpolated ice concentration
% % ice_conc_interp = zeros(length(lon_ceres), length(lat_ceres), 303);
% % 
% % k = 1;
% % for i = 253:555  % Process files 253 to 555
% %     fprintf('Processing file %d of %d: %s\n', i, length(list), list(i).name);
% % 
% %     % Read ice concentration data
% %     ice_conc = ncread(sprintf('%s/%s',list(i).folder,list(i).name),'ice_conc');
% % 
% %     % Handle NaN values and create valid data mask
% %     valid_mask = ~isnan(ice_conc) & ~isnan(lat_conc) & ~isnan(lon_conc_360) & ice_conc >= 0;
% % 
% %     if sum(valid_mask(:)) > 0  % Check if there's valid data
% %         % Create interpolant using valid data points
% %         F = scatteredInterpolant(lon_conc_360(valid_mask), lat_conc(valid_mask), ...
% %                                 ice_conc(valid_mask), 'linear', 'none');
% % 
% %         % Interpolate to CERES grid
% %         test = F(LON_CERES, LAT_CERES)';
% %         ice_conc_interp(:,:,k)  = test;
% %         % % Optional: Clean up interpolated data
% %         % ice_conc_interp(ice_conc_interp(:,:,k) < 0, k) = 0;      % Remove negative values
% %         % ice_conc_interp(ice_conc_interp(:,:,k) > 100, k) = 100;  % Cap at 100%
% %     else
% %         % If no valid data, fill with NaN
% %         ice_conc_interp(:,:,k) = NaN(size(LON_CERES));
% %         warning('No valid ice concentration data in file %d', i);
% %     end
% % 
% %     k = k + 1;
% % end
% 
% list = dir('Month_SIC/ice_conc_nh_ease2-*04.nc');
% for i = 1:41
%     % Read ice concentration data
%     ice_conc = ncread(sprintf('%s/%s',list(i+3).folder,list(i+3).name),'ice_conc');
% 
%     % Handle NaN values and create valid data mask
%     valid_mask = ~isnan(ice_conc) & ~isnan(lat_conc) & ~isnan(lon_conc_360) & ice_conc >= 0;
% 
%     if sum(valid_mask(:)) > 0  % Check if there's valid data
%         % Create interpolant using valid data points
%         F = scatteredInterpolant(lon_conc_360(valid_mask), lat_conc(valid_mask), ...
%                                 ice_conc(valid_mask), 'linear', 'none');
%         % Interpolate to CERES grid
%         test_SIC = F(lon_360, lat_A);
%         % ice_conc_AVHRR(:,:,i)  = test;
% 
%         test = reshape(SIT_April_M(i,:,:),361,361);
%         % ice_volume_April(i) = nansum(test(find(lat_A>=60 & lat_A<=81.5 & test_SIC>=15 & test_SIC<=100))).*25.*25./1000;
%         ice_volume_April(i) = nansum(nansum(test_SIC(find(lat_A>=40 & lat_A<=81.5)).*test(find(lat_A>=40 & lat_A<=81.5)))).*25.*25./1000./100;
% 
%         % test = reshape(SIT_July_August_M(i,:,:),361,361);
%         % % ice_volume_April(i) = nansum(test(find(lat_A>=60 & lat_A<=81.5 & test_SIC>=15 & test_SIC<=100))).*25.*25./1000;
%         % ice_volume_April(i) = nansum(nansum(test_SIC(find(lat_A>=60 & lat_A<=81.5)).*test(find(lat_A>=60 & lat_A<=81.5)))).*25.*25./1000./100;
% 
%         test1 = reshape(SIT_July_August_M(i,:,:),361,361);
%         test2 = reshape(ALBEDO_July_August_M(i,:,:),361,361);
%         test2(find(isnan(test1))) = NaN;
%         Albedo_July_Aug(i) = nanmean(test2(find(lat_A>=60 & lat_A<=81.5)));
% 
%     end   
% %     % ice_volume(i);
% % 
% end
% 
% % fp = fopen('AVHRR_ice_volume.txt','wt');
% % for i = 1:numel(ice_volume_April)
% %     fprintf(fp, '%d %.3f\n', YY(i), ice_volume_April(i));
% % end
% % 
% % fclose(fp);
% % 
% % fp = fopen('AVHRR_albedo.txt','wt');
% % for i = 1:numel(Albedo_July_Aug)
% %     fprintf(fp, '%d %.3f\n', YY(i), Albedo_July_Aug(i));
% % end
% % 
% % fclose(fp);
% 
% 
% % for i = 1:size(albedo,3)
% %     Test1 = albedo(:,:,i);
% %     Test2 = ice_conc_interp(:,:,i);
% %     Albedo_mean(i) = nanmean(Test1(find(Test2>15 & Test2<=100)));
% % end
% % 
% % YYYY = linspace(2000,2024,25);
% % for i = 1:numel(YYYY)
% %     albedo_mean(i) = nanmean(Albedo_mean(5+(YYYY(i)-2000)*12:6+(YYYY(i)-2000)*12));
% % end
% % 
% % fp = fopen('CERES_albedo.txt','wt');
% % for i = 1:numel(albedo_mean)
% %     fprintf(fp, '%d %.3f\n', YYYY(i), albedo_mean(i));
% % end
% % 
% % fclose(fp);