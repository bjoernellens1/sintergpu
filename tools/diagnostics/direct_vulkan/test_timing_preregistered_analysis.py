# SPDX-License-Identifier: Apache-2.0
import pytest
from timing_preregistered_analysis import analyze,percentile95

def run(old=100,candidate=90,after=100,one_outlier=False):
    samples=[]
    for i in range(50):
        for s,val in enumerate((old,candidate,after)):
            if one_outlier and s==1 and i<4:val=200
            samples.append({'sample':i,'slot':s,'ticks':(0,1,val+1,val+10),'transfer_ns':val,'host_bound_ns':1000})
    return {'timestamp_valid_bits':32,'period_ns':1.,'samples':samples}

def test_three_runs_good_deterministic_and_unqualified():
    a=analyze([run(),run(),run()]);assert a==analyze([run(),run(),run()])
    assert all(a['descriptive_nonregression_flags'].values()) and a['descriptive_pipeline_fusion_flag']
    assert a['scope'].startswith('UNQUALIFIED')

def test_drift_p95_and_worse_or_missing_runs_fail():
    a=analyze([run(after=120),run(),run()]);assert not a['descriptive_nonregression_flags']['transfer_host_ns']
    a=analyze([run(one_outlier=True),run(),run()]);assert not a['descriptive_nonregression_flags']['transfer_host_ns']
    a=analyze([run(candidate=101),run(),run()]);assert not a['descriptive_pipeline_fusion_flag']
    with pytest.raises(ValueError):analyze([run(),run()])
    broken=run();broken['samples'][1]['slot']=2
    with pytest.raises(ValueError,match='ABA order'):analyze([broken,run(),run()])

def test_nearest_rank_p95():
    assert percentile95(list(range(1,51)))==48
    with pytest.raises(ValueError):percentile95([0]*50)


def test_fusion_requires_absolute_median_improvement_and_strict_positive():
    mixed=run()
    for sample in mixed['samples']:
        i=sample['sample'];s=sample['slot'];base=1 if i<25 else 100;value=(0.5 if i<25 else 99) if s==1 else base
        # Integer ticks use 2x scaling; compute and pipeline interval are identical.
        sample['ticks']=(0,0,int(value*2),int(value*2))
    result=analyze([mixed,mixed,mixed]);assert result['runs'][0]['resident_pipeline_ns']['paired_median_relative_change']<-.05
    assert not result['descriptive_pipeline_fusion_flag']
    tied=run(old=100,candidate=100,after=100);assert not analyze([tied,tied,tied])['descriptive_pipeline_fusion_flag']
