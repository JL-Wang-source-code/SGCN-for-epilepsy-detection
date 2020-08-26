function [adjmatrix] = LPVG(time_series,L)
[num,len,~] = size(time_series);
time_series = abs(fft(time_series))/len;
min_value=min(time_series);
if (min_value<0)
    time_series = time_series+abs(min_value);
end
adjmatrix = zeros(len,len);
xlmatrix = zeros(len,len);
for i = 1:len
    for j = i:len
        xlmatrix(i,j)=(time_series(j)-time_series(i))/abs(i-j);
    end
end
min_matrix=min(min(xlmatrix));
for k = 1:len
    Y = ones(1,L+1);
    Y = min_matrix*Y;
    for l = k+1:len
        if abs(k-l)<=L+1
            adjmatrix(k,l)=1;adjmatrix(l,k)=1;
            if Y(1,1)<xlmatrix(k,l) 
               Y(1,1)=xlmatrix(k,l);
               Y=sort(Y);
            end   
        elseif Y(1,1)<xlmatrix(k,l)
            adjmatrix(k,l)=1;adjmatrix(l,k)=1;
            Y(1,1)=xlmatrix(k,l);
            Y=sort(Y);
        else    
            adjmatrix(k,l)=0;adjmatrix(l,k)=0;
        end
    end
end
