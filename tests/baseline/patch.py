import re,sys,os
m={'gradient':'gradient_hjh','gauss_newton':'Gauss_Newton_hjh_re','gauss_newton_fd':'Gauss_Newton_delta_hjh','lm_v1':'LM_hjh','lm_v2':'LM_hjh_2','cg':'CG_hjh','pcg':'PCG_hjh','lsqnonlin':'Jall_FINAL'}
for name,suf in m.items():
    L=open(f'orig/EDRegisterSparse_Show_iter_{suf}.m').read().split('\n')
    out=[]
    for ln,s in enumerate(L,1):
        t=s.strip()
        if ln==1:
            out.append(s)
            out.append("global BL_ERR BL_LAM BL_LAMOUT BL_ELAPSED BL_INIT; BL_ERR=[]; BL_LAM=[]; BL_LAMOUT=[]; BL_ELAPSED=NaN; BL_INIT=[]; %BASELINE")
            continue
        if suf=='Jall_FINAL' and re.match(r'^% \[(src|tar)_face|^% rng\(1|^% SrcMesh\.Vpoi|^% TarMesh\.Vpoi',s):
            out.append(s[2:]+" %BASELINE uncommented"); continue
        if suf=='Jall_FINAL' and re.match(r'^(Src|Tar)Mesh = load\(fullfile',s):
            out.append('% '+s+" %BASELINE commented"); continue
        if re.match(r'^show_iteration = true;',t):
            out.append(s.replace('true','false')+" %BASELINE (no pause(1)/figures per iteration)"); continue
        if re.match(r'^show_final = true;',t):
            out.append(s.replace('true','false')+" %BASELINE"); continue
        if t=='toc;' and suf!='Jall_FINAL':
            out.append("    BL_ELAPSED = toc; %BASELINE"); continue
        if s.startswith('function P0 = runOptim'):
            out.append("save('"+name+".mat','BL_ERR','BL_LAM','BL_LAMOUT','BL_ELAPSED','P0','deformedV','nodeIdx','n_verts','n_nodes','-v7'); %BASELINE" if False else "%BASELINE_SAVE")
        out.append(s)
        if s.startswith('function P0 = runOptim'):
            out.append("    global BL_ERR BL_LAM BL_LAMOUT BL_ELAPSED BL_INIT; %BASELINE")
        if re.match(r'^\s*fprintf\("Iter %d Error',s):
            out.append("        BL_ERR(end+1,1) = sum(F.^2); fprintf('BL iter %d err %.15g\\n', i, BL_ERR(end)); %BASELINE")
            if 'used_lambda' in s:
                out.append("        BL_LAM(end+1,1) = used_lambda; BL_LAMOUT(end+1,1) = lambda; fprintf('BL iter %d lambda_used %.15g lambda_out %.15g\\n', i, used_lambda, lambda); %BASELINE")
            elif 'lambda' in s:
                out.append("        BL_LAM(end+1,1) = lambda; BL_LAMOUT(end+1,1) = lambda; %BASELINE")
    txt='\n'.join(out)
    save=f"""BL_P0 = P0; BL_deformedV = gather(deformedV); BL_nodeIdx = nodeIdx; %BASELINE
BL_iters = numel(BL_ERR); BL_matlab = version; %BASELINE
save('{name}.mat','BL_ERR','BL_LAM','BL_LAMOUT','BL_ELAPSED','BL_INIT','BL_iters','BL_matlab','BL_P0','BL_deformedV','BL_nodeIdx'); %BASELINE
fprintf('BL done: %d iters, elapsed %.3f s\\n', BL_iters, BL_ELAPSED); %BASELINE
"""
    txt=txt.replace("%BASELINE_SAVE",save)
    txt=re.sub(r"(?<![A-Za-z_])gpuArray\(","bl_gpuArray(",txt)
    if suf=='Jall_FINAL':
        txt=txt.replace("    function stop = outfun(x, optimValues, state)\n","    function stop = outfun(x, optimValues, state)\n        if strcmp(state,'init'), BL_INIT = optimValues.resnorm; end %BASELINE\n        if strcmp(state,'iter'), BL_ERR(end+1,1) = optimValues.resnorm; fprintf('BL iter %d resnorm %.15g\\n', optimValues.iteration, optimValues.resnorm); end %BASELINE\n")
        txt=txt.replace("    P0 = lsqnonlin(","    tic; %BASELINE\n    P0 = lsqnonlin(",1)
        txt=txt.replace("    disp('done!');\n    \n    function stop","    BL_ELAPSED = toc; %BASELINE\n    disp('done!');\n    \n    function stop",1)
        txt=txt.replace("function P0 = runOptim(P0, SrcMesh","function P0 = runOptim(P0, SrcMesh",1)
    open(f'patched/{name}.m','w').write(txt)
