function[] = create_supereeg_electrode_map(data_fname, subj, enum, outfile)

x = load(data_fname);
locs = cellfun(@str2num, x.struct.R, 'UniformOutput', false);
R = cat(1, locs{:});

colors = zeros(size(R)); %black
inds = find(strcmpi(subj, x.struct.Subject));
colors(inds, 1) = 1; %red
colors(inds(enum), :) = [0 0 1]; %blue

sizes = 10.*ones(1, size(R, 1));
sizes(inds) = 50;
sizes(inds(enum)) = 75;

save(outfile, 'R', 'colors', 'sizes');

