# @Author XI RAO
# CITS4001 Research Project
# These are the helper functions for findings the bins for the datasets features, aim to reduce the search space for the pso training
# This feature discretisation method is originally applied in LightGBM.


def GreedyFindBin(
    distinct_values, 
    counts,num_distinct_values, 
    max_bin, total_cnt, 
    min_data_in_bin=3
    ):

    bin_upper_bound=[]
    assert(max_bin>0)       # Avoid the fist bin is 0
    
    if num_distinct_values <= max_bin:
        cur_cnt_inbin = 0
        for i in range(num_distinct_values-1):
            cur_cnt_inbin += counts[i]
            if cur_cnt_inbin >= min_data_in_bin:
                bin_upper_bound.append((distinct_values[i] + distinct_values[i + 1]) / 2.0)
                cur_cnt_inbin = 0
        cur_cnt_inbin += counts[num_distinct_values - 1]
        bin_upper_bound.append(float('Inf'))
        
    else:
        if min_data_in_bin>0:
            max_bin=min(max_bin,total_cnt//min_data_in_bin)
            max_bin=max(max_bin,1)

        mean_bin_size=total_cnt/max_bin
        rest_bin_cnt = max_bin
        rest_sample_cnt = total_cnt

        is_big_count_value=[False]*num_distinct_values
        for i in range(num_distinct_values):
            if counts[i] >= mean_bin_size:
                is_big_count_value[i] = True
                rest_bin_cnt-=1
                rest_sample_cnt -= counts[i]

        mean_bin_size = rest_sample_cnt/rest_bin_cnt
        upper_bounds=[float('Inf')]*max_bin
        lower_bounds=[float('Inf')]*max_bin
        
        bin_cnt = 0
        lower_bounds[bin_cnt] = distinct_values[0]        
        cur_cnt_inbin = 0

        for i in range(num_distinct_values-1):
            if not is_big_count_value[i]:
                rest_sample_cnt -= counts[i]        
            cur_cnt_inbin += counts[i]
            
            if is_big_count_value[i] or cur_cnt_inbin >= mean_bin_size or \
            is_big_count_value[i + 1] and cur_cnt_inbin >= max(1.0, mean_bin_size * 0.5):
                upper_bounds[bin_cnt] = distinct_values[i] 
                bin_cnt+=1
                lower_bounds[bin_cnt] = distinct_values[i + 1]
                if bin_cnt >= max_bin - 1:
                    break

                cur_cnt_inbin = 0
                if not is_big_count_value[i]:
                    rest_bin_cnt-=1
                    mean_bin_size = rest_sample_cnt / rest_bin_cnt
        bin_cnt+=1
        for i in range(bin_cnt-1):
            bin_upper_bound.append((upper_bounds[i] + lower_bounds[i + 1]) / 2.0)
        bin_upper_bound.append(float('Inf'))
    return bin_upper_bound


def FindBinWithZeroAsOneBin(
    distinct_values, 
    counts,
    num_distinct_values, 
    max_bin, 
    total_cnt, 
    min_data_in_bin=3
    ):

    bin_upper_bound=list()
    assert(max_bin>0)

    left_cnt_data = 0
    cnt_zero = 0
    right_cnt_data = 0
    kZeroThreshold = 1e-35
    for i in range(num_distinct_values):
        if distinct_values[i] <= -kZeroThreshold:
            left_cnt_data += counts[i]
        elif distinct_values[i] > kZeroThreshold:
            right_cnt_data += counts[i]
        else:
            cnt_zero += counts[i]
 
    left_cnt = -1
    for i in range(num_distinct_values):
        if distinct_values[i] > -kZeroThreshold:
            left_cnt = i
            break
    
    if left_cnt < 0:
        left_cnt = num_distinct_values

    if left_cnt > 0:
        left_max_bin = int( left_cnt_data/ (total_cnt - cnt_zero) * (max_bin - 1) )
        left_max_bin = max(1, left_max_bin)
        bin_upper_bound = GreedyFindBin(
            distinct_values, 
            counts, left_cnt, 
            left_max_bin, 
            left_cnt_data, 
            min_data_in_bin
            )
        bin_upper_bound[-1] = -kZeroThreshold

    right_start = -1
    for i in range(left_cnt, num_distinct_values):
        if distinct_values[i] > kZeroThreshold:
            right_start = i
            break

    if right_start >= 0:
        right_max_bin = max_bin - 1 - len(bin_upper_bound)
        assert(right_max_bin>0)
        right_bounds = GreedyFindBin(
            distinct_values[right_start:], 
            counts[right_start:],
            num_distinct_values - right_start, 
            right_max_bin, 
            right_cnt_data, 
            min_data_in_bin
            )
        bin_upper_bound.append(kZeroThreshold)
        bin_upper_bound+=right_bounds
    else:
        bin_upper_bound.append(float('Inf'))

    return bin_upper_bound


def GetBins(df,col_names, max_bin, min_data_in_bin=3):
    bins={}
    def _count(arr):
        distinct_values=[arr[0]]
        counts=[]
        counts_dict={arr[0]:1}
        for i in range(1,len(arr)):
            if arr[i]==arr[i-1]:
                counts_dict[arr[i]]+=1
            else:
                distinct_values.append(arr[i])
                counts_dict[arr[i]]=1

        for x in distinct_values:
            counts.append(counts_dict[x])

        return distinct_values, counts
        
    for col in col_names:
        tmp=df[col].to_list()
        tmp.sort()
        distinct_values, counts = _count(tmp)
        num_distinct_values=len(distinct_values)
        total_cnt=sum(counts)
        bins[col]=FindBinWithZeroAsOneBin(
            distinct_values, 
            counts, 
            num_distinct_values, 
            max_bin, total_cnt, 
            min_data_in_bin=min_data_in_bin
            )

    return bins
