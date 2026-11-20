rng(0); mx = 0; mxR = 0;
rs = [randn(20,3); 1e-1*randn(5,3); 1e-3*randn(5,3); 1e-6*randn(5,3); 0.049*[1 0 0]; 0.051*[1 0 0]; 0 0 0; 3*[0 0 1]/1];
for k = 1:size(rs,1)
    r = rs(k,:); [R,J] = cv.Rodrigues(r);
    mxR = max(mxR, norm(R'*R-eye(3),'fro')); mxR = max(mxR, abs(det(R)-1));
    Jn = zeros(3,9); h = 1e-6;
    for i=1:3
        d=zeros(1,3); d(i)=h;
        Rp = cv.Rodrigues(r+d); Rm = cv.Rodrigues(r-d);
        Jn(i,:) = reshape(((Rp-Rm)/(2*h)).',1,9);
    end
    e = max(abs(J(:)-Jn(:))); mx = max(mx,e);
    if k>=size(rs,1)-8, fprintf('r=[%g %g %g] err=%g\n', r, e); end
end
fprintf('max FD error %.3g, max orthogonality err %.3g\n', mx, mxR);
[R,J]=cv.Rodrigues([0 0 0]); disp(R); disp(J);
% knnsearch check
X=randn(500,3); Y=randn(300,3); i1=knnsearch(X,Y,'K',1);
[~,i2]=min(pdist2_(Y,X),[],2); fprintf('knn mismatches %d\n', sum(i1~=i2));
function D=pdist2_(A,B), D=zeros(size(A,1),size(B,1)); for k=1:size(A,1), D(k,:)=sum((B-A(k,:)).^2,2)'; end, end
