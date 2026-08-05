function run_simpoly_oracle(image_path, output_json_path)
    % Standalone MATLAB oracle wrapper for SIMPoly benchmarking
    if nargin < 2
        error('Usage: run_simpoly_oracle(image_path, output_json_path)');
    end

    try
        img = imread(image_path);
        if size(img, 3) > 1
            gray = rgb2gray(img);
        else
            gray = img;
        end

        % 1. CLAHE
        enhanced = adapthisteq(gray, 'NumTiles', [8 8], 'ClipLimit', 0.01);

        % 2. Morphological Reconstruction
        se = strel('disk', 3);
        marker = imerode(enhanced, se);
        reconstructed = imreconstruct(marker, enhanced);

        % 3. Canny Edges
        edges = edge(reconstructed, 'canny');

        % 4. Otsu Binarization
        level = graythresh(reconstructed);
        mask = im2bw(reconstructed, level);

        % 5. Morphological Cleaning
        mask_clean = bwareaopen(mask, 50);
        mask_closed = imclose(mask_clean, strel('disk', 2));
        mask_filtered = medfilt2(mask_closed, [3 3]);
        mask_dilated = imdilate(mask_filtered, strel('disk', 1));

        % 6. Skeleton & Distance Transform
        skel = bwskel(mask_dilated);
        edt_map = bwdist(~mask_dilated);

        % 7. Local Diameters
        skel_pts = find(skel);
        diameters = 2 * edt_map(skel_pts);

        if isempty(diameters)
            mean_val = 0;
            median_val = 0;
            gauss_center = 0;
            gauss_sigma = 0;
        else
            mean_val = mean(diameters);
            median_val = median(diameters);
            [counts, bin_centers] = histcounts(diameters, 30);
            centers = (bin_centers(1:end-1) + bin_centers(2:end)) / 2;
            [~, max_idx] = max(counts);
            gauss_center = centers(max_idx);
            gauss_sigma = std(diameters);
        end

        result = struct();
        result.run_id = 'MATLAB_ORACLE_RUN';
        result.oracle_id = 'SIMPOLY_MATLAB_ORIGINAL';
        result.oracle_version = '1.0.0';
        result.image_path = image_path;
        result.gaussian_center_px = gauss_center;
        result.gaussian_sigma_px = gauss_sigma;
        result.arithmetic_mean_px = mean_val;
        result.median_px = median_val;
        result.status = 'SUCCESS';

        json_text = jsonencode(result);
        fid = fopen(output_json_path, 'w');
        fprintf(fid, '%s', json_text);
        fclose(fid);

    catch exc
        err_struct = struct();
        err_struct.status = 'FAILED';
        err_struct.message = exc.message;
        json_text = jsonencode(err_struct);
        fid = fopen(output_json_path, 'w');
        fprintf(fid, '%s', json_text);
        fclose(fid);
    end
end
