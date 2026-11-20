function cfg = defaultConfig()
%DEFAULTCONFIG  Default parameters of the Embedded Deformation registration.
%
%   cfg = edsm.defaultConfig()
%
%   Output (struct, scalar fields):
%     nVertexNeighbor : number of ED nodes that influence each mesh vertex (4)
%     nNodeNeighbor   : number of neighbors per ED node in the smoothness term (6)
%                       (geodesic mode: this count includes the node itself)
%     sampleRate      : fraction of source vertices used as ED nodes (0.01)
%     wRot            : weight of the rotation terms (1.0)
%     wSmooth         : weight of the smoothness term (0.1)
%     wLap            : weight of the Laplacian term (0.01)
%     distanceOnMesh  : true = geodesic (graph) distances, false = Euclidean (true)
%     legacyJacobian  : false = corrected analytic Jacobian (default); true reproduces
%                       the Jacobian of the original scripts bit-for-bit, including its
%                       known errors (rot1/rot2 column order, Laplacian r columns,
%                       invalid-neighbor vertices). F is the same in both modes. (false)
%     seed            : seed passed to rng(seed,'twister') before node sampling (1)
%
%   The values equal the hard-coded settings of the EDRegisterSparse_Show_iter_*
%   scripts.

cfg = struct( ...
    'nVertexNeighbor', 4, ...
    'nNodeNeighbor',   6, ...
    'sampleRate',      0.01, ...
    'wRot',            1.0, ...
    'wSmooth',         0.1, ...
    'wLap',            0.01, ...
    'distanceOnMesh',  true, ...
    'legacyJacobian',  false, ...
    'seed',            1);
end
