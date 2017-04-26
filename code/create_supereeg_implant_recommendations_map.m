function[] = create_supereeg_implant_recommendations_map(fname, outfile1, outfile2, tau, theta)


data = load(fname);
if ~exist('tau', 'var'), tau = 1; end
if ~exist('theta', 'var'), theta = 1; end

std_fname = 'MNI152_T1_2mm_brain.nii';
[~, Rstd, origin, ~, mask] = TFA_load_nii(std_fname);

R = cellfun(@str2num, data.struct.R, 'UniformOutput', false);
R = cat(1, R{:});

corrs = data.struct.Correlation;
subjs = data.struct.Subject;

good_inds = ~isnan(corrs);
corrs = corrs(good_inds);
R = R(good_inds, :);
subjs = subjs(good_inds);

unique_subjs = unique(subjs);
subj_inds = cellfun(@(x)(find(strcmpi(x, unique_subjs), 1, 'first')), subjs);

%v1: color each electrode by the average reconstruction correlation observed
%for the patient who contributed that electrode.  then interpolate to get
%the full map.
subj_zcorrs = zeros(size(unique_subjs));
[c, w] = deal(zeros(size(corrs)));
for i = 1:length(unique_subjs)
    next_inds = subj_inds == i;
    subj_zcorrs(i) = mean(r2z(corrs(next_inds)));
    
    c(next_inds) = c(next_inds) + subj_zcorrs(i);
    w(next_inds) = w(next_inds) + 1;
end
c = z2r(c ./ w);
d = pdist2(R, Rstd);
s = exp(-tau.*d);

y1 = (c * s) ./ sum(s, 1);
cmu_to_nii(y1, Rstd, false, outfile1, mask, origin);

%v2: for each electrode, color the electrode by the average correlation
%observed for all patients with electrodes at most theta units (euclidean
%distance) away.
expected_zcorrs = nan([1 size(R, 1)]);
d = pdist2(R, R);
for i = 1:size(R, 1)
    next_inds = d(:, i) <= theta;
    if sum(next_inds) >= 1
        expected_zcorrs(i) = mean(subj_zcorrs(subj_inds(next_inds)));
    end
end

y2 = (z2r(expected_zcorrs) * s) ./ sum(s, 1);
cmu_to_nii(y2, Rstd, false, outfile2, mask, origin);

