function s = insert_events(s, idx, event, varargin)
% This function takes a continuous sequence (e.g. sound) `s` and adds a
% given `event` at all positions given by their indices `idx`. It can
% either insert the `event` while erasing whatever was present in `s`
% before, or it can sum the event with whatever was there before. 
%
% Parameters
% ----------
% s : array_like, shape=[1,N]
%     Continous sequence. 
% idx : array_like, int
%     Vector of index positions. The waveform in `event` will be repeatedly
%     pasted into `s` starting from samples given in elements of `idx`.
% event : array_like, shape=[1,N_event]
%     Continous waveform of the unitary event that will be pasted into `s`.
% vol : array_like, float between 0 and 1, optional
%     The gain factor (volume) applied to `event` before insertion. 
% add : bool, optional, default=false
%     If true, the event will be summed with whatever was already present
%     in `s` at the given position. 
% 
% Returns 
% -------
% s : array_like, shape=[1,N]
%     Continous sequence with inserted events. 
%

parser = inputParser; 

addParameter(parser, 'vol', []); 
addParameter(parser, 'add', false); 

parse(parser, varargin{:}); 

vol = parser.Results.vol; 
add_sound = parser.Results.add; 

if ~isempty(vol) && length(vol) ~= length(idx)
   error('%d volume values but %d events', length(vol), length(idx));   
end

%%

if ~iscell(event)
    event = {event}; 
end

curr_event_idx = repmat([1:length(event)], ...
                        1, ceil(length(idx)/length(event))); 

for i=1:length(idx)
    
    if isempty(vol)
        gain = 1; 
    else
        gain = vol(i); 
    end
    
    curr_event = event{curr_event_idx(i)}; 
    
    if add_sound
        s(idx(i)+1 : idx(i)+length(curr_event)) = ...
            s(idx(i)+1 : idx(i)+length(curr_event)) + gain * curr_event; 
    else
        s(idx(i)+1 : idx(i)+length(curr_event)) = gain * curr_event; 
    end
    
end