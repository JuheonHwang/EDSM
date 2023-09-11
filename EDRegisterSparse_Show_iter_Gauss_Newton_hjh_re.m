clear; close all; fclose all;
DIR = 'Dataset7CResult';
Src = 'mat_mesh_020';
Tar = 'mat_mesh_022';

warning('off', 'MATLAB:MKDIR:DirectoryExists'); warning('off', 'MATLAB:rmpath:DirNotFound');
rmpath('./opencv/mexopencv/'); addpath('./opencv/mexopencv/');
rmpath('./opencv/mexopencv/opencv_contrib/'); addpath('./opencv/mexopencv/opencv_contrib/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iteration = 100;
step_size = 0.001;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_vertex_neighbor = 4;
EDSampleRate = 0.01;
w_rot = 1.0; %100;
w_smooth = 0.1; %1000;
w_lap = 0.01;
distanceOnMesh = true;
show_iteration = true;
show_final = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n_node_neighbor = 6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
assert(n_vertex_neighbor > 1);
assert(n_node_neighbor > 1);

%opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'UseParallel', true); %off
%opts = optimoptions(@lsqnonlin, 'Display', 'iter', 'Algorithm', 'trust-region-reflective',...
%    'SpecifyObjectiveGradient', true, 'FunctionTolerance', 1e-10);

[src_face, src_pts] = jwutils.plyread('simple_poisson_mesh_020.ply', 'tri');
[tar_face, tar_pts] = jwutils.plyread('simple_poisson_mesh_022.ply', 'tri');
rng(1, 'twister')
SrcMesh.Vpoi = src_pts; SrcMesh.Fpoi = src_face;
TarMesh.Vpoi = tar_pts; TarMesh.Fpoi = tar_face;

% SrcMesh = load(fullfile(DIR, Src));
% TarMesh = load(fullfile(DIR, Tar));

pntSrc = pointCloud(SrcMesh.Vpoi);
pntTar = pointCloud(TarMesh.Vpoi);

n_verts = size(SrcMesh.Vpoi, 1);
n_nodes = round(n_verts * EDSampleRate);

nodeIdx = randperm(n_verts, n_nodes);

pntEDNodes = pointCloud(SrcMesh.Vpoi(nodeIdx, :));

fprintf("The number of points of source : %d\n", n_verts);
fprintf("The number of points of target : %d\n", size(TarMesh.Vpoi, 1));
fprintf("The number of points of ED Nodes : %d\n", n_nodes);

SrcMesh.Npoi = vertexNormal(triangulation(SrcMesh.Fpoi, SrcMesh.Vpoi));
TarMesh.Npoi = vertexNormal(triangulation(TarMesh.Fpoi, TarMesh.Vpoi));


R0 = eye(3); T0 = [0, 0, 0];
A0 = repmat(reshape(eye(3), [3 3 1]), [1 1 n_nodes]);
t0 = zeros(3, n_nodes);
P0 = encodeParam(R0, T0, A0, t0);
G0 = mean(SrcMesh.Vpoi, 1);

if distanceOnMesh
    s = [SrcMesh.Fpoi(:, 1); SrcMesh.Fpoi(:, 2); SrcMesh.Fpoi(:, 3)];
    t = [SrcMesh.Fpoi(:, 2); SrcMesh.Fpoi(:, 3); SrcMesh.Fpoi(:, 1)];
    w = sqrt(sum((SrcMesh.Vpoi(s, :) - SrcMesh.Vpoi(t, :)).^2, 2));
    G = graph(s, t, w);
    G = simplify(G);
    Lap = laplacian(G);
    fprintf('Get Distances... ');
    D = distances(G);
    D = D(:, nodeIdx);
    fprintf('Get Nearests... ');
    [distNeighbor, idxNeighbor] = mink(D, n_vertex_neighbor+1, 2);
    distNeighbor = distNeighbor'; idxNeighbor = idxNeighbor';
    D = D(nodeIdx, :);
    [distNodeNeighbor, idxNodeNeighbor] = mink(D, n_node_neighbor, 2);
    idxNodeNeighbor = idxNodeNeighbor';
    assert(~any(isnan(distNodeNeighbor(:))));
    
    distNeighbor = 1.0 - (distNeighbor ./ distNeighbor(n_vertex_neighbor+1, :));
    idxNeighbor = idxNeighbor(1:n_vertex_neighbor, :); distNeighbor = distNeighbor(1:n_vertex_neighbor, :);
else
    [idxNeighbor, distNeighbor] = multiQueryKNNSearchImpl(pntEDNodes, pntSrc.Location, n_vertex_neighbor+2); %#ok<*UNRCH>
    idxNodeNeighbor = multiQueryKNNSearchImpl(pntEDNodes, pntEDNodes.Location, n_node_neighbor+1);
    distNeighbor = sqrt(distNeighbor);
    
    distNeighbor = 1.0 - (distNeighbor ./ distNeighbor(n_vertex_neighbor+2, :));
    idxNeighbor = idxNeighbor(2:(1+n_vertex_neighbor), :); distNeighbor = distNeighbor(2:(1+n_vertex_neighbor), :);
    idxNodeNeighbor = idxNodeNeighbor(2:(1+n_node_neighbor), :);
end

distNeighbor = distNeighbor ./ sum(distNeighbor, 1); % Normalize Dist Weights

invalidIdx = any(isnan(distNeighbor), 1);
distNeighbor(:, invalidIdx) = 1 / n_node_neighbor;
idxNeighbor(:, invalidIdx) = NaN;


dispMesh(SrcMesh.Vpoi, SrcMesh.Fpoi, -90, 0);
dispMesh(TarMesh.Vpoi, TarMesh.Fpoi, -90, 0);
gpuSrcMesh = SrcMesh;
gpuSrcMesh.Vpoi = gpuArray(gpuSrcMesh.Vpoi);
gpuSrcMesh.Npoi = gpuArray(gpuSrcMesh.Npoi);
pntTar.Normal = TarMesh.Npoi;

fprintf('Run Optim... \n');
%P0 = lsqnonlin(@(P)EfuncLocal(P, gpuSrcMesh, pntTar, gpuArray(G0), nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, gpuArray(Lap), w_rot, w_smooth, w_lap), P0, [], [], opts);
P0 = runOptim(P0, iteration, gpuSrcMesh, pntTar, TarMesh.Fpoi, gpuArray(G0), nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, gpuArray(Lap), w_rot, w_smooth, w_lap, R0, T0, A0, t0, show_iteration, show_final);

[R0, T0, A0, t0] = decodeParam(P0, n_nodes);

v0 = SrcMesh.Vpoi';
g0 = v0(:, nodeIdx);
n0 = SrcMesh.Npoi';

deformedV = deformED(v0, n0, g0, A0, t0, G0, R0, T0, idxNeighbor, distNeighbor);


dispMesh(deformedV, SrcMesh.Fpoi, -90, 0);
dispMesh(SrcMesh.Vpoi, SrcMesh.Fpoi, -90, 0);
dispMesh(TarMesh.Vpoi, TarMesh.Fpoi, -90, 0);

jwutils.saveMesh(deformedV, SrcMesh.Fpoi, -90, 0, 'Front.png');
jwutils.saveMesh(deformedV, SrcMesh.Fpoi, -90, 180, 'Back.png');

function P0 = runOptim(P0, iteration, SrcMesh, pntTar, faceTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, R_, T_, A_, t_, show_iteration, show_final)
    tic;
    epsilon = 0.000001;
    epsilon2 = 0.001;
    epsilon3 = 0.01;
    
    disp_lims = [];
    F_prev= [];
    
    for i = 1:iteration

        [P0, F, Gradient, delta] = EfuncLocal(P0, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, R_, T_, A_, t_);
        fprintf("Iter %d Error : %f \n", i, sum(F.^2));

        if show_iteration || (i == iteration && show_final)
            [R0, T0, A0, t0] = decodeParam(P0, size(idxNodeNeighbor, 2));
            
            v0 = SrcMesh.Vpoi';
            n0 = SrcMesh.Npoi';
            g0 = v0(:, nodeIdx);
            deformedV = deformED(v0, n0, g0, A0, t0, G0, R0, T0, idxNeighbor, distNeighbor);
            
            disp_lims = dispMeshes(gather(deformedV), gather(SrcMesh.Fpoi), pntTar.Location, faceTar, -90, 0, disp_lims);
            pause(1)
        end
        
        if (~isempty(F_prev) & (abs(F - F_prev) < (epsilon * (1 + F)))) |  (norm(Gradient, inf) < (epsilon3 * (1 + F))) | (norm(delta, inf) < (epsilon2 * (1 + norm(delta, inf))))
            break
        end
        
        F_prev = F;
    end

    toc;
    disp('done!');
end



function [P0, F, Gradient, delta] = EfuncLocal(P0, SrcMesh, pntTar, G0, nodeIdx, idxNeighbor, idxNodeNeighbor, distNeighbor, Lap, w_rot, w_smooth, w_lap, R_, T_, A_, t_, indices)  
    n_nodes = length(nodeIdx);
    
    [R0, T0, A0, t0, J_R0] = decodeParam(P0, n_nodes); % J_R0 : 3 x 9
    [R0, T0, A0, t0] = replaceParam(R0, T0, A0, t0, R_, T_, A_, t_);
    
    v0 = SrcMesh.Vpoi';
    g0 = v0(:, nodeIdx);
    
    n_verts = size(distNeighbor, 2);
    n_neighbor = size(distNeighbor, 1);
    assert(n_verts > n_neighbor);
    
    distNeighbor_ = reshape(distNeighbor', [1 size(distNeighbor')]);
    invalidNeibor = isnan(idxNeighbor);
    idxNeighbor(invalidNeibor) = 1;
    
    g1 = reshape(g0(:, idxNeighbor'), [size(g0, 1) size(idxNeighbor')]);
    t1 = reshape(t0(:, idxNeighbor'), [size(t0, 1) size(idxNeighbor')]);
    A1 = reshape(A0(:, :, idxNeighbor'), [size(A0, [1 2]) size(idxNeighbor')]);
    v1 = reshape(v0 - g1, [size(g1, 1) 1 size(g1, 2:3)]);
    v1_trans = squeeze(pagemtimes(A1, v1)) + g1 + t1;
    
        
    deformedV_local = squeeze(sum(distNeighbor_ .* v1_trans, 3))' - G0(:)'; % Local Transform %% JW FIXED
    deformedV = deformedV_local * R0' + T0(:)' + G0(:)'; % Global Transform %% JW FIXED
    deformedV(any(invalidNeibor, 1), :) = v0(:, any(invalidNeibor, 1))';
    
    if ~exist('indices', 'var')
        indices = knnsearch(gather(deformedV), pntTar.Location, 'K', 1);
    end
    
    Efit = (gather(deformedV(indices, :)) - pntTar.Location) ./ sqrt(size(pntTar.Location, 1));

    Elap = gather((Lap * deformedV) ./ full(diag(Lap)));
    
    Erot1 = (pagemtimes(A0, 'transpose', A0, 'none') - reshape(eye(3), 3, 3, 1)) ./ sqrt(n_nodes);
    Erot2 = (pagedet(A0) - 1.0) ./ sqrt(n_nodes);
    
    g_node1 = reshape(g0(:, idxNodeNeighbor'), [size(g0, 1) size(idxNodeNeighbor')]);
    t_node1 = reshape(t0(:, idxNodeNeighbor'), [size(t0, 1) size(idxNodeNeighbor')]);
    v_node1 = reshape(g_node1 - g0, [size(g_node1, 1) 1 size(g_node1, 2:3)]);
    Esmooth = gather(sum(squeeze(pagemtimes(A0, v_node1)) + g0 + t0 - g_node1 - t_node1, 3)) ./ sqrt(3 * n_nodes);
    
    F = [Efit(:); w_rot*Erot1(:); w_rot*Erot2(:); w_smooth*Esmooth(:); w_lap* Elap(:)];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Jacobian%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    J_R0 = sparse(J_R0(:, [1 4 7 2 5 8 3 6 9]));

    n_target = length(pntTar.Location);
    nA = 9 * n_nodes;

    lap = gather(Lap);

    deformedV_R_col = reshape(repmat([1 4 7 2 5 8 3 6 9], [n_verts 1]), [], 1);
    deformedV_R_row = reshape(repmat(reshape(1:3*n_verts, [], 3), [3, 1]), [], 1);
    deformedV_R_val = repmat(reshape(gather(deformedV_local), [], 1), [3 1]);
    
    JdeformedV_R = sparse(deformedV_R_row, deformedV_R_col, deformedV_R_val, numel(Elap), 9) * J_R0'; % JW FIX

    % deformedV_T_col = reshape(repmat(1:3, [n_verts 1]), [], 1);
    % deformedV_T_row = reshape(1:3*n_verts, [], 1);
    % deformedV_T_val = ones(3*n_verts, 1);
    % JdeformedV_T = sparse(deformedV_T_row, deformedV_T_col, deformedV_T_val, numel(Elap), 3);

    deformedV_A_col = reshape(repmat(reshape(repmat(9 * (reshape(idxNeighbor, 1, []) - 1), [9 1]) + [1; 2; 3; 4; 5; 6; 7; 8; 9], 1, []), [3 1]), [], 1);
    deformedV_A_row = reshape(repmat(reshape(1:3*n_verts, [], 3)', [3 * 3 * n_neighbor, 1]), [], 1);
    deformedV_A_val = repelem(reshape(permute(distNeighbor_ .* squeeze(v1), [1 3 2]), 3, []), 9, 1);
    deformedV_A_val = gather(repmat(R0(:), [3 1]) .* deformedV_A_val);

    JdeformedV_A = sparse(deformedV_A_row, deformedV_A_col, deformedV_A_val(:), numel(Elap), nA);

    deformedV_t_col = repelem(reshape(repmat(3 * (reshape(idxNeighbor, 1, []) - 1), [3 1]) + [1; 2; 3], [], 1), 3, 1);
    deformedV_t_row = reshape(repmat(reshape(1:3*n_verts, [], 3)', [3 * n_neighbor, 1]), [], 1);
    deformedV_t_val = reshape(R0(:) .* repelem(reshape(squeeze(permute(distNeighbor_, [1 3 2])), 1, []), 9, 1), [], 1);
    % deformedV_t_val = reshape(R0(:) .* repelem(reshape(distNeighbor, 1, []), 9, 1), [], 1);

    JdeformedV_t = sparse(deformedV_t_row, deformedV_t_col, deformedV_t_val, numel(Elap), 3 * n_nodes);

    Jfit_T = spalloc(numel(Efit), 3, numel(Efit));
    Jfit_T(1:n_target, 1) = 1;
    Jfit_T((n_target + 1):(2 * n_target), 2) = 1;
    Jfit_T((2 * n_target + 1):(3 * n_target), 3) = 1;

    % Jfit_At = JdeformedV_At([indices n_verts+indices 2*n_verts+indices], :);
    % Jfit_R = JdeformedV_R([indices n_verts+indices 2*n_verts+indices], :);
    % Jfit = [Jfit_R, Jfit_T, Jfit_At];
    JdeformedV_At = [JdeformedV_A, JdeformedV_t];
    Jfit = [JdeformedV_R([indices n_verts+indices 2*n_verts+indices], :), Jfit_T, JdeformedV_At([indices n_verts+indices 2*n_verts+indices], :)];

    %JdeformedV = [JdeformedV_R, JdeformedV_T, JdeformedV_A, JdeformedV_t]; % JW FIX
    % Jfit_origin = JdeformedV([indices n_verts+indices 2*n_verts+indices], :);
    % Jlap_origin = [lap*JdeformedV(1:n_verts, :);
    %                lap*JdeformedV(n_verts+(1:n_verts), :);
    %                lap*JdeformedV(2*n_verts+(1:n_verts), :)]; % JW FIX

    Jlap_At = [lap*JdeformedV_At(1:n_verts, :);
        lap*JdeformedV_At(n_verts+(1:n_verts), :);
        lap*JdeformedV_At(2*n_verts+(1:n_verts), :)];
    Jlap = [sparse(numel(Elap), 6), Jlap_At];

    smooth_A_row = reshape(repmat(reshape(1:numel(Esmooth), 3, []), [n_neighbor - 1, 1]), [], 1);
    smooth_A_col = reshape((1:nA), [], 1);
    smooth_A_val = gather(reshape(repmat(reshape(squeeze(sum(v_node1, 4)), 1, []), [n_neighbor - 1, 1]), [], 1));

    Jsmooth_A = sparse(smooth_A_row, smooth_A_col, smooth_A_val, numel(Esmooth), nA);

    smooth_t_row = reshape(repmat(1:3*n_nodes, [size(idxNodeNeighbor, 1), 1]), [], 1);
    %smooth_t_col = reshape([3 * idxNodeNeighbor - 2; 3 * idxNodeNeighbor - 1; 3 * idxNodeNeighbor], [], 1);
    smooth_t_col = repmat(3 * idxNodeNeighbor, [3 1]) + repmat(repelem([-2; -1; 0], size(idxNodeNeighbor, 1)), [1 n_nodes]);
    t_val = repmat(-1, [size(idxNodeNeighbor, 1), 1]);
    t_val(1) = size(idxNodeNeighbor, 1) - 1;
    smooth_t_val = repmat(t_val, [3*n_nodes, 1]);

    Jsmooth_t = sparse(smooth_t_row, smooth_t_col, smooth_t_val, numel(Esmooth), 3 * n_nodes);

    Jsmooth = [sparse(numel(Esmooth), 6), Jsmooth_A, Jsmooth_t];


    rot1_row =[1 2 3 4 7; 2 4 5 6 8; 3 6 7 8 9; 1 2 3 4 7; 2 4 5 6 8; 3 6 7 8 9; 1 2 3 4 7; 2 4 5 6 8; 3 6 7 8 9]';
    rot1_rows = repmat(reshape(rot1_row, [], 1), [1 n_nodes])  + 9 * ((1:n_nodes) - 1);

    rot1_cols = reshape(reshape(repmat((1:9) + 6, [5 n_nodes]), [], n_nodes) + repmat(9 * ((1:n_nodes) - 1), [45 1]), [], 1);

    A = reshape(permute(A0, [2 1 3]), [], size(A0, 3));
    rot1_vals = [2 * A(1, :); A(2, :); A(3, :); A(2, :); A(3, :);
        A(1, :); A(1, :); 2 * A(2, :); A(3, :); A(3, :);
        A(1, :); A(2, :); A(1, :); A(2, :); 2 * A(3, :);
        2 * A(4, :); A(5, :); A(6, :); A(5, :); A(6, :);
        A(4, :); A(4, :); 2 * A(5, :); A(6, :); A(6, :);
        A(4, :); A(5, :); A(4, :); A(5, :); 2 * A(6, :);
        2 * A(7, :); A(8, :); A(9, :); A(8, :); A(9, :);
        A(7, :); A(7, :); 2 * A(8, :); A(9, :); A(9, :);
        A(7, :); A(8, :); A(7, :); A(8, :); 2 * A(9, :)];
    rot1_vals = reshape(rot1_vals, [], 1);

    Jrot1 = sparse(rot1_rows, rot1_cols, rot1_vals, numel(Erot1), numel(P0));

    rot2_rows = reshape(repmat(1:n_nodes, [9 1]), [], 1);

    rot2_cols = reshape(reshape(repmat((1:9) + 6, [1 n_nodes]), [], n_nodes) + repmat(9 * ((1:n_nodes) - 1), [9 1]), [], 1);

    rot2_vals = [A(5, :) .* A(9, :) - A(6, :) .* A(8, :);
        A(6, :) .* A(7, :) - A(4, :) .* A(9, :);
        A(4, :) .* A(8, :) - A(5, :) .* A(7, :);
        A(3, :) .* A(8, :) - A(2, :) .* A(9, :);
        A(1, :) .* A(9, :) - A(3, :) .* A(7, :);
        A(2, :) .* A(7, :) - A(1, :) .* A(8, :);
        A(2, :) .* A(6, :) - A(3, :) .* A(5, :);
        A(3, :) .* A(4, :) - A(1, :) .* A(6, :);
        A(1, :) .* A(5, :) - A(2, :) .* A(4, :)];
    rot2_vals = reshape(rot2_vals, [], 1);

    Jrot2 = sparse(rot2_rows, rot2_cols, rot2_vals, numel(Erot2), numel(P0));

    J = [Jfit ./ sqrt(n_target);
        w_rot * Jrot1 ./ sqrt(n_nodes);
        w_rot * Jrot2 ./ sqrt(n_nodes);
        w_smooth * Jsmooth ./ sqrt(3 * n_nodes);
        w_lap * Jlap ./ reshape(repmat(full(gather(diag(Lap))), [3 1]), [], 1)]; %% JW FIXED NEEDED CHECK

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Gauss-Newton%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Gradient = 2 * (F' * J)';
    % delta = -(inv(J'*J)*J' * F);
    % delta = -((J'*J)\J' * F);
    delta = lsqminnorm(J, -F);
    P0 = P0 + delta;

end

function [deformedV, deformedN] = deformED(v0, n0, g0, A0, t0, G0, R0, T0, idxNeighbor, distNeighbor)
    n_verts = size(distNeighbor, 2);
    n_neighbor = size(distNeighbor, 1);
    assert(n_verts > n_neighbor);
    
    distNeighbor = reshape(distNeighbor', [1 size(distNeighbor')]);
    invalidNeibor = isnan(idxNeighbor);
    idxNeighbor(invalidNeibor) = 1;
    
    g1 = reshape(g0(:, idxNeighbor'), [size(g0, 1) size(idxNeighbor')]);
    t1 = reshape(t0(:, idxNeighbor'), [size(t0, 1) size(idxNeighbor')]);
    A1 = reshape(A0(:, :, idxNeighbor'), [size(A0, [1 2]) size(idxNeighbor')]);
    v1 = reshape(v0 - g1, [size(g1, 1) 1 size(g1, 2:3)]);
    v1 = squeeze(pagemtimes(A1, v1)) + g1 + t1;
    
    n0 = gpuArray(reshape(n0, [size(n0, 1) 1 size(n0, 2)]));
    
    if nargout > 1
        deformedN = squeeze(sum(squeeze(pagemtimes(pagefun(@transpose, pagefun(@inv, gpuArray(A1))), n0)) .* distNeighbor, 3))';
    end
    
    deformedV = squeeze(sum(distNeighbor .* v1, 3))'; % Local Transform
    deformedV = (deformedV - G0(:)') * R0' + T0(:)' + G0(:)'; % Global Transform
    deformedV(any(invalidNeibor, 1), :) = v0(:, any(invalidNeibor, 1))';
end

function D = pagedet(X)
    D = X(1, 1, :).*X(2, 2, :).*X(3, 3, :) + X(1, 2, :).*X(2, 3, :).*X(3, 1, :) + X(1, 3, :).*X(2, 1, :).*X(3, 2, :) ...
        - X(1, 3, :).*X(2, 2, :).*X(3, 1, :) - X(1, 2, :).*X(2, 1, :).*X(3, 3, :) - X(1, 1, :).*X(2, 3, :).*X(3, 2, :);
end

function dispMesh(V, F, rot1, rot2)
    figure; 
    jwutils.dispMesh(V, F, [0.8 0.8 0.8 1.0]);
    camorbit(rot1, 0, 'data', [0 0 1]);
    camorbit(rot2, 0, 'data ', [1 0 0]);
    axis off;
end

function lims = dispMeshes(V1, F1, V2, F2, rot1, rot2, lims)
    figure; 
    jwutils.dispMesh(V1, F1, [0.5 0.2 0.2 0.3]);
    jwutils.dispMesh(V2, F2, [0.2 0.5 0.2 0.3]);
    camorbit(rot1, 0, 'data', [0 0 1]);
    camorbit(rot2, 0, 'data ', [1 0 0]);
    axis off;
    if nargin > 6
        if isempty(lims)
            lims = [xlim; ylim; zlim];
        else
            xlim(lims(1, :)); ylim(lims(2, :)); zlim(lims(3, :));
        end
    end
end

function P = encodeParam(R, T, A, t)
    n_verts = size(A, 3);
    assert(size(t, 2) == n_verts);
    P = [rotationMatrixToVector(R) T(:)', A(:)', t(:)']';
end

function [R, T, A, t, J_R] = decodeParam(P, n_verts)
    P = P(:);
    [R, J_R] = cv.Rodrigues(P(1:3)');
    T = P(4:6)';
    nA = 9*n_verts;
    A = P(7:(7+nA-1));
    t = P((7+nA):end);
    A = reshape(A, [3, 3, n_verts]);
    t = reshape(t, [3, n_verts]);
end

function [R0, T0, A0, t0] = replaceParam(R0, T0, A0, t0, R_, T_, A_, t_)
    if isempty(R0)
        R0 = R_;
    end
   
    if isempty(T0)
        T0 = T_;
    end

    if isempty(A0)
        A0 = A_;
    end

    if isempty(t0)
        t0 = t_;
    end
end