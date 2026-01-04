function d=disteusq(x,y,mode,w)
% DISTEUSQ 유클리디안, 자승 유클리디안 혹은 마하나노비스 거리를 계산한다. D=(X,Y,MODE,W)
%
% 입력 인자: 
%  X,Y    비교될 벡터집합. 각 열이 데이터 벡터로 되어 있다. 
%         X와 Y는 같은 수의 행을 가진다.  
%
%  MODE   옵션 사항 선택 문자열:
%         'x' 모든 X와 Y의 열에서 전체 거리 행렬 계산. 
%         'd' 관련된 X와 Y의 열간의 거리만 계산. 
%             만약 X와 Y가 같은 열의 수를 가지면 디폴트로 'd'이고 그렇지 않으면 'x'.
%         's' 유클리디안 거리 결과의 자승근을 취한다. 
%
%  W      옵션 가중치 행렬: 계산된 거리는 (x-y)*W*(x-y)'
%         만약 W가 벡터이면 행렬 diag(W)가 사용된다. 
%           
% 출력 인자: 
%  D      만약 MODE='d'이면 D는 X와 Y보다 작은 같은 수의 열을 가진 행 벡터이다. 
%         만역 MODE='x'이면 D는 X와 같은 수의 열과 Y'와 같은 수의 행을 가진 행렬이다. 
%
%   VOICEBOX 공개소스 수정
%   VOICEBOX home page: http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.html
%
[nx,p]=size(x); ny=size(y,1);
if nargin<3 | isempty(mode) mode='0'; end
if any(mode=='d') | (mode~='x' & nx==ny)
   nx=min(nx,ny);
   z=x(1:nx,:)-y(1:nx,:);
   if nargin<4
      d=sum(z.*conj(z),2);
   elseif min(size(w))==1
      wv=w(:).';
      d=sum(z.*wv(ones(size(z,1),1),:).*conj(z),2);
   else
      d=sum(z*w.*conj(z),2);
   end
else
   if p>1
      if nargin<4
         z=permute(x(:,:,ones(1,ny)),[1 3 2])-permute(y(:,:,ones(1,nx)),[3 1 2]);
         d=sum(z.*conj(z),3);
      else
         nxy=nx*ny;
         z=reshape(permute(x(:,:,ones(1,ny)),[1 3 2])-permute(y(:,:,ones(1,nx)),[3 1 2]),nxy,p);
         if min(size(w))==1
            wv=w(:).';
            d=reshape(sum(z.*wv(ones(nxy,1),:).*conj(z),2),nx,ny);
         else
            d=reshape(sum(z*w.*conj(z),2),nx,ny);
         end
      end
   else
      z=x(:,ones(1,ny))-y(:,ones(1,nx)).';
      if nargin<4
         d=z.*conj(z);
      else
         d=w*z.*conj(z);
      end
   end
end
if any(mode=='s')
   d=sqrt(d);
end

