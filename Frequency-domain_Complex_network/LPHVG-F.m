function adjmatrix = LPHVG(time_series,L)
[num,len,~] = size(time_series);
time_series = abs(fft(time_series))/len;
min_value=min(time_series);
if (min_value<0)
    time_series = time_series+abs(min_value);
end
[num,len,~] = size(time_series);
adjmatrix = zeros(len,len);
for k = 1:len
    Y = zeros(1,L+1);
    for l = k+1:len
        if abs(k-l)<=L+1
            adjmatrix(k,l)=1;adjmatrix(l,k)=1;
            if Y(1,1)<time_series(l)
               Y(1,1)=time_series(l);
               Y=sort(Y);
            end
        elseif time_series(k)>Y(1,1) && time_series(l)>Y(1,1)
            adjmatrix(k,l)=1;adjmatrix(l,k)=1;
            Y(1,1)=time_series(l);
            Y=sort(Y);
        end
    end
end
