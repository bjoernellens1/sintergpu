# SPDX-License-Identifier: Apache-2.0
"""Preregistered exploratory paired analysis; never grants qualification.

Three separate process runs, exactly 50 validated ABA triples per run. GPU time
uses stage timestamps, not a dispatch-exclusive clock. Bootstrap uses fixed length-five circular blocks to retain short-range serial
dependence; longer dependence and stationarity remain unverified. Monitor,
source, device, lock and environmental receipts are verified elsewhere.
"""
import math
import random
import statistics

BOOTSTRAP_REPLICATES=5000
SEED=0x53494E544552
BLOCK_LENGTH=5
ALPHA=0.025/9
DRIFT_LIMIT=0.05
P95_REGRESSION_LIMIT=0.05
PIPELINE_IMPROVEMENT_MIN=0.05
METRICS=('compute_stage_ns','resident_pipeline_ns','transfer_host_ns')


def percentile95(data):
    # Preregistered nearest rank; for 50 observations choose order statistic 48.
    if len(data)!=50 or any(not math.isfinite(x) or x<=0 for x in data): raise ValueError('50 positive finite samples required')
    return sorted(data)[math.ceil(0.95*len(data))-1]


def _paired(a,b,c,rng):
    if any(len(x)!=50 for x in (a,b,c)):raise ValueError('exactly 50 ABA triples')
    for values in (a,b,c):percentile95(values)
    # Dimensionless paired differences; old controls bracket each candidate.
    ratios=[b[i]/((a[i]+c[i])/2)-1 for i in range(50)]
    median=statistics.median(ratios);baseline=statistics.median((a[i]+c[i])/2 for i in range(50))
    def resample():
        values=[]
        for _ in range(50//BLOCK_LENGTH):
            start=rng.randrange(50);values.extend(ratios[(start+j)%50] for j in range(BLOCK_LENGTH))
        return statistics.median(values)
    boot=sorted(resample() for _ in range(BOOTSTRAP_REPLICATES))
    upper=boot[math.ceil((1-ALPHA)*len(boot))-1]
    ma=statistics.median(a);mc=statistics.median(c);drift=abs(mc-ma)/((ma+mc)/2)
    pa=percentile95(a);pb=percentile95(b);pc=percentile95(c)
    return {'old_before_median_ns':ma,'candidate_median_ns':statistics.median(b),'old_after_median_ns':mc,
            'paired_median_relative_change':median,'paired_baseline_median_ns':baseline,'paired_block_bootstrap_upper_bonferroni':upper,
            'old_control_median_relative_drift':drift,'old_before_p95_ns':pa,'candidate_p95_ns':pb,'old_after_p95_ns':pc,
            'control_drift_within_5_percent':drift<=DRIFT_LIMIT,
            'candidate_p95_within_5_percent_of_both_controls':pb<=min(pa,pc)*(1+P95_REGRESSION_LIMIT),
            'bootstrap_upper_nonpositive':upper<=0,'bootstrap_upper_strictly_negative':upper<0}


def analyze(runs):
    """Compute descriptive gates only; input identities/controls remain unverified."""
    if len(runs)!=3:raise ValueError('exactly three process runs')
    rng=random.Random(SEED);output=[]
    for run in runs:
        if len(run['samples'])!=150:raise ValueError('150 ordered samples required')
        bits=run['timestamp_valid_bits'];period=run['period_ns']
        if type(bits)!=int or not 1<=bits<=64 or type(period) not in (int,float) or not math.isfinite(period) or period<=0:raise ValueError('timestamp capability')
        mask=(1<<bits)-1
        values={metric:[[],[],[]] for metric in METRICS}
        for index,sample in enumerate(run['samples']):
            if sample['sample']!=index//3 or sample['slot']!=index%3:raise ValueError('ABA order')
            t0,t1,t2,t3=sample['ticks'];slot=index%3;bound=sample['host_bound_ns']
            if type(bound)!=int or bound<=0 or bound>=math.ldexp(period,bits) or any(type(t)!=int or t<0 or t>mask for t in (t0,t1,t2,t3)):raise ValueError('timestamp bound')
            if any(not ((b-a)&mask) or ((b-a)&mask)*period>bound for a,b in ((t0,t3),(t1,t2))):raise ValueError('timestamp interval')
            values['compute_stage_ns'][slot].append(((t2-t1)&mask)*period)
            values['resident_pipeline_ns'][slot].append(((t3-t0)&mask)*period)
            values['transfer_host_ns'][slot].append(sample['transfer_ns'])
        output.append({metric:_paired(*values[metric],rng) for metric in METRICS})
    def passed(r,m):return r[m]['control_drift_within_5_percent'] and r[m]['candidate_p95_within_5_percent_of_both_controls'] and r[m]['bootstrap_upper_nonpositive']
    return {'policy':{'bootstrap_replicates':BOOTSTRAP_REPLICATES,'seed':SEED,'percentile':'nearest rank',
            'confidence':'one-sided 1-(0.025/9) percentile circular block bootstrap of median paired relative change; family has 3 metrics x 3 runs',
            'block_length':BLOCK_LENGTH,'family_alpha':0.025,
            'control_drift_limit':DRIFT_LIMIT,'p95_regression_limit':P95_REGRESSION_LIMIT,
            'pipeline_median_improvement_min':PIPELINE_IMPROVEMENT_MIN},'runs':output,
            'descriptive_nonregression_flags':{m:all(passed(r,m) for r in output) for m in METRICS},
            'descriptive_pipeline_fusion_flag':all(passed(r,'resident_pipeline_ns') and r['resident_pipeline_ns']['bootstrap_upper_strictly_negative'] and r['resident_pipeline_ns']['candidate_median_ns']<=r['resident_pipeline_ns']['paired_baseline_median_ns']*(1-PIPELINE_IMPROVEMENT_MIN) for r in output),
            'scope':'UNQUALIFIED descriptive computation; no verified controls, independent runs, long-range dependence/stationarity validation, hardware qualification or adoption decision'}
