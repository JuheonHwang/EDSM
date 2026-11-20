function [V, F] = readPly(path)
%READPLY  Minimal PLY reader returning vertices and triangle faces.
%
%   [V, F] = readPly(path)
%
%   V : n x 3 double (x, y, z of element 'vertex')
%   F : m x 3 double, 1-based, from the list property of element 'face'
%       ('vertex_indices' or 'vertex_index').

fid = fopen(path, 'r');
if fid < 0, error('edsm:readPly', 'Cannot open %s', path); end
cleaner = onCleanup(@() fclose(fid));

% ---- header -------------------------------------------------------------
assert(strcmp(strtrim(fgetl(fid)), 'ply'), 'Not a PLY file.');
fmt = '';
elems = struct('name', {}, 'count', {}, 'props', {});
while true
    line = fgetl(fid);
    if ~ischar(line), error('edsm:readPly', 'Missing end_header.'); end
    tok = strsplit(strtrim(line));
    switch lower(tok{1})
        case 'format'
            fmt = lower(tok{2});
        case 'element'
            elems(end+1) = struct('name', tok{2}, 'count', str2double(tok{3}), 'props', struct('name', {}, 'type', {}, 'countType', {})); %#ok<AGROW>
        case 'property'
            if strcmp(tok{2}, 'list')
                p = struct('name', tok{5}, 'type', tok{4}, 'countType', tok{3});
            else
                p = struct('name', tok{3}, 'type', tok{2}, 'countType', '');
            end
            elems(end).props(end+1) = p;
        case 'end_header'
            break
    end
end

% ---- body ---------------------------------------------------------------
switch fmt
    case 'ascii'
        stream = struct('kind', 'ascii', 'data', fscanf(fid, '%f'), 'pos', 1);
    case {'binary_little_endian', 'binary_big_endian'}
        stream = struct('kind', fmt, 'data', fread(fid, Inf, '*uint8'), 'pos', 1);
    otherwise
        error('edsm:readPly', 'Unsupported PLY format ''%s''.', fmt);
end

V = [];
F = [];
for e = 1:numel(elems)
    el = elems(e);
    [vals, lists, stream] = readElement(el, stream);
    if strcmp(el.name, 'vertex')
        names = {el.props.name};
        V = [vals(:, strcmp(names, 'x')), vals(:, strcmp(names, 'y')), vals(:, strcmp(names, 'z'))];
    elseif strcmp(el.name, 'face')
        names = {el.props.name};
        k = find(ismember(names, {'vertex_indices', 'vertex_index'}), 1);
        assert(~isempty(k), 'edsm:readPly', 'Face element has no vertex index list.');
        F = fanTriangulate(lists{k});
    end
end
V = double(V);
F = double(F);
end

function T = fanTriangulate(L)
% L : cell (m x 1), each a 0-based index row vector. Output m' x 3, 1-based.
m = numel(L);
T = zeros(m, 3);
extra = zeros(0, 3);
for k = 1:m
    f = L{k};
    T(k, :) = f(1:3);
    for j = 4:numel(f)
        extra(end+1, :) = [f(1), f(j-1), f(j)]; %#ok<AGROW>
    end
end
T = [T; extra] + 1;
end

function [vals, lists, s] = readElement(el, s)
% vals  : count x nScalarProps matrix (columns of list properties are empty)
% lists : cell per property, {count x 1} cells of row vectors for list properties
np = numel(el.props);
n = el.count;
vals = zeros(n, np);
lists = cell(np, 1);
isList = ~cellfun(@isempty, {el.props.countType});
for k = find(isList), lists{k} = cell(n, 1); end

if strcmp(s.kind, 'ascii')
    for i = 1:n
        for k = 1:np
            if ~isList(k)
                vals(i, k) = s.data(s.pos);
                s.pos = s.pos + 1;
            else
                c = s.data(s.pos);
                lists{k}{i} = s.data(s.pos + (1:c))';
                s.pos = s.pos + 1 + c;
            end
        end
    end
    return
end

bigEndian = strcmp(s.kind, 'binary_big_endian');
if ~any(isList)
    % Fixed stride: decode each property from a strided byte matrix.
    sizes = cellfun(@typeSize, {el.props.type});
    stride = sum(sizes);
    block = reshape(s.data(s.pos:(s.pos + n * stride - 1)), stride, n);
    off = 0;
    for k = 1:np
        vals(:, k) = decode(block((off + 1):(off + sizes(k)), :), el.props(k).type, bigEndian);
        off = off + sizes(k);
    end
    s.pos = s.pos + n * stride;
    return
end

% Elements containing lists: sequential cursor.
for i = 1:n
    for k = 1:np
        p = el.props(k);
        if ~isList(k)
            sz = typeSize(p.type);
            vals(i, k) = decode(s.data(s.pos:(s.pos + sz - 1)), p.type, bigEndian);
            s.pos = s.pos + sz;
        else
            csz = typeSize(p.countType);
            c = decode(s.data(s.pos:(s.pos + csz - 1)), p.countType, bigEndian);
            s.pos = s.pos + csz;
            isz = typeSize(p.type);
            raw = reshape(s.data(s.pos:(s.pos + c * isz - 1)), isz, c);
            lists{k}{i} = decode(raw, p.type, bigEndian)';
            s.pos = s.pos + c * isz;
        end
    end
end
end

function sz = typeSize(t)
switch t
    case {'char', 'uchar', 'int8', 'uint8'},      sz = 1;
    case {'short', 'ushort', 'int16', 'uint16'},  sz = 2;
    case {'int', 'uint', 'int32', 'uint32', 'float', 'float32'}, sz = 4;
    case {'double', 'float64'},                   sz = 8;
    otherwise, error('edsm:readPly', 'Unknown PLY type ''%s''.', t);
end
end

function v = decode(bytes, t, bigEndian)
% bytes : sz x k uint8 (each column one value). Returns k x 1 double.
switch t
    case {'char', 'int8'},      cls = 'int8';
    case {'uchar', 'uint8'},    cls = 'uint8';
    case {'short', 'int16'},    cls = 'int16';
    case {'ushort', 'uint16'},  cls = 'uint16';
    case {'int', 'int32'},      cls = 'int32';
    case {'uint', 'uint32'},    cls = 'uint32';
    case {'float', 'float32'},  cls = 'single';
    case {'double', 'float64'}, cls = 'double';
end
if bigEndian
    bytes = flipud(bytes);
end
v = double(typecast(bytes(:), cls));
v = v(:);
end
