function x = bl_gpuArray(x)
% Shim: no GPU available; identity. (builtin gpuArray cannot be shadowed by path, so patched scripts call bl_gpuArray)
end
