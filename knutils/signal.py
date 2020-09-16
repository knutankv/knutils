function [ datai ] = time_int(data,fs,levels,domain, filters, postfilter)
%TIME_INT Inegrate input data specified number of levels.
%
% INPUT:    data            Data matrix / vector
%           fs              Sample rate
%           levels          Number of times to perform integration.
%           domain          'freq' or 'time'    (time is standard)
%           filters         Cell array with filter objects arranged to correspond
%                           to different integration levels. If empty, no
%                           filtering is applied.
%           postfilter      Apply filter post as well. true / false
%           
%
% OUTPUT:   datai           Integrated data
%
% Use like this: [ datai ] = time_int(data,fs,levels)
%
%
% Knut Andreas Kvaale, 2017
%

thisdata = data;
L = size(data,1);
t = 0:1/fs:(L-1)*(1/fs);
f = [fs*(0:floor((L-1)/2))/L,-(fs*(ceil((L-1)/2):-1:1)/L)]';

if length(filters) <= 1
    for i=1:levels
        filts{i} = filters;
    end
    filters = filts;
end

for level = 1:levels
    if ~isempty(filters{level})
        filtdata = filtfilt(filters{level},thisdata);
    else
        filtdata = thisdata;
    end
    
    if strcmp(domain,'freq')
        thisdataf = repmat(1./(2*pi*f*1i),[1,size(filtdata,2)]).*fft(filtdata);
        thisdataf(1,:) = 0;
        datai{level} = real(ifft(thisdataf));
    elseif strcmp(domain,'time')
        datai{level} = cumtrapz(t,filtdata);
    end
    
    if postfilter == true && ~isempty(filters{level})
        datai{level} = filtfilt(filters{level},datai{level});
    end
    thisdata = datai{level};
end


end

