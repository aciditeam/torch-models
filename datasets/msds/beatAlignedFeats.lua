----
-- Million Song Dataset Getters
-- Beat Aligned Features
-- 
-- TODO:
--  * Partial implementation, complete,
--  * No support for duration getter (!),
--  * Tensors are converted to doubleTensors.
--   Check if floatTensors wouldn't be faster.
----
--
-- ADAPTED FROM Python code by:
-- Thierry Bertin-Mahieux (2011) Columbia University
-- tb2332@columbia.edu

-- Code to get beat-aligned features (chromas or timbre)
-- from the HDF5 song files of the Million Song Dataset.

-- This is part of the Million Song Dataset project from
-- LabROSA (Columbia University) and The Echo Nest.


-- Copyright 2011, Thierry Bertin-Mahieux
-- parts of this code from Ron J. Weiss

-- This program is free software: you can redistribute it and/or modify
-- it under the terms of the GNU General Public License as published by
-- the Free Software Foundation, either version 3 of the License, or
-- (at your option) any later version.

-- This program is distributed in the hope that it will be useful,
-- but WITHOUT ANY WARRANTY; without even the implied warranty of
-- MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
-- GNU General Public License for more details.

-- You should have received a copy of the GNU General Public License
-- along with this program.  If not, see <http://www.gnu.org/licenses/>.

bt_getters = require './millionSongGetters'

torch.setdefaulttensortype('torch.FloatTensor')

function bt_getters.get_btchromas(h5)
   -- Get beat-aligned chroma from a song file of the Million Song Dataset
   -- The first dimension indexes time
   -- INPUT:
   --    h5          - filename or open h5 file
   -- RETURN:
   --    btchromas   - beat-aligned chromas, one beat per row
   --                  or None if something went wrong (e.g. no beats)
   --
   -- if string, open and get chromas, if h5, get chromas
   if type(h5) == 'string' then
      h5 = bt_getters.open_h5_file_read(h5)
   end

   chromas = bt_getters.get_segments_pitches(h5)
   segstarts = bt_getters.get_segments_start(h5):float()
   btstarts = bt_getters.get_beats_start(h5):float()
   -- duration = bt_getters.get_duration(h5)

   -- get the series of starts for segments and beats by flattening matrices
   -- NOTE: MAYBE USELESS?
   -- result for track: 'TR0002Q11C3FA8332D'
   --    segstarts.shape = (708,)
   --    btstarts.shape = (304,)
   segstarts = segstarts:view(segstarts:numel())
   btstarts = btstarts:view(btstarts:numel())
   -- aligned features
   btchroma = align_feats(chromas:t(), segstarts, btstarts) --, duration)
   if not btchroma then
      return nil
   end
   -- Actually don't normalize here. TODO: do it in the network.
   -- Renormalize. Each column max is 1.
   -- maxs = (btchroma:max(1))
   -- maxs:eq(0): -- [np.where(maxs == 0)] = 1
   -- btchroma = (btchroma / maxs)
   -- Done.
   return btchroma:t()
end

--[[
def get_btchromas_loudness(h5):
    """
    Similar to btchroma, but adds the loudness back.
    We use the segments_loudness_max
    There is no max value constraint, simply no negative values.
    """
    # if string, open and get chromas, if h5, get chromas
    if type(h5).__name__ == 'str':
        h5 = bt_getters.open_h5_file_read(h5)
        chromas = bt_getters.get_segments_pitches(h5)
        segstarts = bt_getters.get_segments_start(h5)
        btstarts = bt_getters.get_beats_start(h5)
        duration = bt_getters.get_duration(h5)
        loudnessmax = bt_getters.get_segments_loudness_max(h5)
        h5.close()
    else:
        chromas = bt_getters.get_segments_pitches(h5)
        segstarts = bt_getters.get_segments_start(h5)
        btstarts = bt_getters.get_beats_start(h5)
        duration = bt_getters.get_duration(h5)
        loudnessmax = bt_getters.get_segments_loudness_max(h5)
    # get the series of starts for segments and beats
    segstarts = np.array(segstarts).flatten()
    btstarts = np.array(btstarts).flatten()
    # add back loudness
    chromas = chromas.T * idB(loudnessmax)
    # aligned features
    btchroma = align_feats(chromas, segstarts, btstarts, duration)
    if btchroma is None:
        return None
    # done (no renormalization)
    return btchroma


def get_bttimbre(h5):
    """
    Get beat-aligned timbre from a song file of the Million Song Dataset
    INPUT:
       h5          - filename or open h5 file
    RETURN:
       bttimbre    - beat-aligned timbre, one beat per column
                     or None if something went wrong (e.g. no beats)
    """
    # if string, open and get timbre, if h5, get timbre
    if type(h5).__name__ == 'str':
        h5 = bt_getters.open_h5_file_read(h5)
        timbre = bt_getters.get_segments_timbre(h5)
        segstarts = bt_getters.get_segments_start(h5)
        btstarts = bt_getters.get_beats_start(h5)
        duration = bt_getters.get_duration(h5)
        h5.close()
    else:
        timbre = bt_getters.get_segments_timbre(h5)
        segstarts = bt_getters.get_segments_start(h5)
        btstarts = bt_getters.get_beats_start(h5)
        duration = bt_getters.get_duration(h5)
    # get the series of starts for segments and beats
    # NOTE: MAYBE USELESS?
    # result for track: 'TR0002Q11C3FA8332D'
    #    segstarts.shape = (708,)
    #    btstarts.shape = (304,)
    segstarts = np.array(segstarts).flatten()
    btstarts = np.array(btstarts).flatten()
    # aligned features
    bttimbre = align_feats(timbre.T, segstarts, btstarts, duration)
    if bttimbre is None:
        return None
    # done (no renormalization)
    return bttimbre

   
def get_btloudnessmax(h5):
    """
    Get beat-aligned loudness max from a song file of the Million Song Dataset
    INPUT:
       h5             - filename or open h5 file
    RETURN:
       btloudnessmax  - beat-aligned loudness max, one beat per column
                        or None if something went wrong (e.g. no beats)
    """
    # if string, open and get max loudness, if h5, get max loudness
    if type(h5).__name__ == 'str':
        h5 = bt_getters.open_h5_file_read(h5)
        loudnessmax = bt_getters.get_segments_loudness_max(h5)
        segstarts = bt_getters.get_segments_start(h5)
        btstarts = bt_getters.get_beats_start(h5)
        duration = bt_getters.get_duration(h5)
        h5.close()
    else:
        loudnessmax = bt_getters.get_segments_loudness_max(h5)
        segstarts = bt_getters.get_segments_start(h5)
        btstarts = bt_getters.get_beats_start(h5)
        duration = bt_getters.get_duration(h5)
    # get the series of starts for segments and beats
    # NOTE: MAYBE USELESS?
    # result for track: 'TR0002Q11C3FA8332D'
    #    segstarts.shape = (708,)
    #    btstarts.shape = (304,)
    segstarts = np.array(segstarts).flatten()
    btstarts = np.array(btstarts).flatten()
    # reverse dB
    loudnessmax = idB(loudnessmax)
    # aligned features
    btloudnessmax = align_feats(loudnessmax.reshape(1,
                                                    loudnessmax.shape[0]),
                                segstarts, btstarts, duration)
    if btloudnessmax is None:
        return None
    # set it back to dB
    btloudnessmax = dB(btloudnessmax + 1e-10)
    # done (no renormalization)
    return btloudnessmax
]]--

function align_feats(feats, segstarts, btstarts) -- , duration)
   -- MAIN FUNCTION: aligned whatever matrix of features is passed,
   -- one column per segment, and interpolate them to get features
   -- per beat.
   -- Note that btstarts could be anything, e.g. bar starts
   -- INPUT
   --    feats      - matrix of features, one column per segment
   --    segstarts  - segments starts in seconds,
   --                 dim must match feats # cols (flatten ndarray)
   --    btstarts   - beat starts in seconds (flatten ndarray)
   --    duration   - overall track duration in seconds
   -- RETURN
   --    btfeats    - features, one column per beat
   --                 None if there is a problem
   
   -- # sanity check
   -- if feats.shape[0] == 0 or feats.shape[1] == 0 then
   if feats:nDimension() < 2 then
      return nil
   end
   -- if btstarts.shape[0] == 0 or segstarts.shape[0] == 0:
   if (btstarts:numel() == 0 or segstarts:numel() == 0) then
      return nil
   end
   
   -- FEAT PER BEAT
   -- Move segment feature onto a regular grid
   -- result for track: 'TR0002Q11C3FA8332D'
   --    warpmat.shape = (304, 708)
   --    btchroma.shape = (304, 12)
   warpmat = get_time_warp_matrix(segstarts, btstarts) --, duration)
   featchroma = torch.mm(warpmat, feats:t():float()):t()
   if featchroma:nDimension() < 2 then -- sanity check
      return nil
   end
   -- done
   return featchroma
end

function get_time_warp_matrix(segstart_v, btstart_v) --, duration)    
   -- Used by create_beat_synchro_chromagram
   -- Returns a matrix (#beats,#segs)
   -- #segs should be larger than #beats, i.e. many events or segs
   -- happen in one beat.
   -- THIS FUNCTION WAS ORIGINALLY CREATED BY RON J. WEISS (Columbia/NYU/Google)
   
   -- length of beats and segments in seconds
   -- result for track: 'TR0002Q11C3FA8332D'
   --    seglen.shape = (708,)
   --    btlen.shape = (304,)
   --    duration = 238.91546    meaning approx. 3min59s
   local segments_n = segstart_v:numel()
   local beats_n = btstart_v:numel()
   
   -- seglen_v = (torch.cat(segstart_v[{{2, segments_n}}],
   -- 		       torch.Tensor({duration}))) -
   --    segstart_v
   local seglen_v = (torch.cat(segstart_v[{{2, segments_n}}],
			       torch.Tensor({segstart_v[segments_n]}))) -
      segstart_v
   -- btlen_v = (torch.cat(btstart_v[{{2,beats_n}}],
   -- 		      torch.Tensor({duration}))) -
   --    btstart_v
   local btlen_v = (torch.cat(btstart_v[{{2,beats_n}}],
			      torch.Tensor({btstart_v[beats_n]}))) -
      btstart_v
   
   local warpmat_m = torch.zeros(segments_n, beats_n)
   -- iterate over beats (columns of warpmat_m)
   for n = 1, beats_n do
      -- beat start time and end time in seconds
      local start_sec = btstart_v[n]
      local end_sec = start_sec + btlen_v[n]
      -- find first segment that starts after beat starts - 1
      -- try:
      -- start_idx = np.nonzero((segstart_v - start_sec) >= 0)[0][0] - 1
      -- except IndexError:
      -- no segment start after that beat, can happen close
      -- to the end, simply ignore, maybe even break?
      -- (catching faster than ckecking... it happens rarely?)
      local start_after_beat_v = (segstart_v - start_sec):ge(0):nonzero()
      if start_after_beat_v:nDimension() >= 2 then
	 start_idx = (segstart_v - start_sec):ge(0):nonzero()[1][1] - 1
      else
	 break
      end
      -- find first segment that starts after beat ends
      -- segs_after = np.nonzero((segstart_v - end_sec) >= 0)[0]
      local segs_after_v = (segstart_v - end_sec):ge(0):nonzero()
      if segs_after_v:nDimension() == 0 then
	 end_idx = start_idx
      else
	 end_idx = segs_after_v[1][1]
      end
      -- fill col of warpmat_m with 1 for the elem in between
      -- (including start_idx, excluding end_idx)
      warpmat_m[{{start_idx, end_idx}, n}] = 1
      -- if the beat started after the segment, keep the proportion
      -- of the segment that is inside the beat
      warpmat_m[{start_idx, n}] = 1 - ((start_sec - segstart_v[start_idx])
	    / seglen_v[start_idx])
      -- if the segment ended after the beat ended, keep the proportion
      -- of the segment that is inside the beat
      if end_idx - 1 > start_idx then
	 warpmat_m[{end_idx-1, n}] = ((end_sec - segstart_v[end_idx-1])
	       / seglen_v[end_idx-1])
      end
      -- normalize so the 'energy' for one beat is one
      local column_energy = warpmat_m:select(2, n):sum()
      warpmat_m:select(2, n):div(column_energy)
   end
   -- return the transpose, meaning (beats , segs)
   return warpmat_m:t()
end

--[[

def idB(loudness_array):
    """
    Reverse the Echo Nest loudness dB features.
    'loudness_array' can be pretty any numpy object:
    one value or an array
    Inspired by D. Ellis MATLAB code
    """
    return np.power(10., loudness_array / 20.)


def dB(inv_loudness_array):
    """
    Put loudness back in dB
    """
    return np.log10(inv_loudness_array) * 20.

def die_with_usage():
    """ HELP MENU """
    print 'beat_aligned_feats.py'
    print '   by T. Bertin-Mahieux (2011) Columbia University'
    print '      tb2332@columbia.edu'
    print ''
    print 'This code is intended to be used as a library.'
    print 'For debugging purposes, you can launch:'
    print '   python beat_aligned_feats.py <SONG FILENAME>'
    sys.exit(0)

if __name__ == '__main__':

    # help menu
    if len(sys.argv) < 2:
        die_with_usage()

    print '*** DEBUGGING ***'

    # params
    h5file = sys.argv[1]
    if not os.path.isfile(h5file):
        print 'ERROR: file %s does not exist.' % h5file
        sys.exit(0)

    # compute beat chromas
    btchromas = get_btchromas(h5file)
    print 'btchromas.shape =', btchromas.shape
    if np.isnan(btchromas).any():
        print 'btchromas have NaN'
    else:
        print 'btchromas have no NaN'
    print 'the max value is:', btchromas.max()

    # compute beat timbre
    bttimbre = get_bttimbre(h5file)
    print 'bttimbre.shape =', bttimbre.shape
    if np.isnan(bttimbre).any():
        print 'bttimbre have NaN'
    else:
        print 'bttimbre have no NaN'
    print 'the max value is:', bttimbre.max()

]]--

return bt_getters
