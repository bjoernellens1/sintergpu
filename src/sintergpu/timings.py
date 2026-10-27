"""Source-bound timing reference; missing data never becomes a zero duration."""
import copy
import hashlib
from pathlib import Path
import math
import re
import statistics

METRICS=('cpu_reference_wall_ms','gpu_kernel_ms','gpu_pipeline_ms','host_submit_complete_ms','transfer_inclusive_ms')


def dependency_identities():
    """Conservative closure: all packaged shaders and independent references."""
    root=Path(__file__).resolve().parent
    def digest(paths):
        h=hashlib.sha256()
        for p in sorted(paths):h.update(p.relative_to(root).as_posix().encode()+b'\0'+p.read_bytes()+b'\0')
        return h.hexdigest()
    return {'shader_tree_sha256':digest((root/'kernels').rglob('*.slang')),
            'reference_tree_sha256':digest(list((root/'fixtures').rglob('*.py'))+list((root/'references').rglob('*.py')))}


def summarize(samples):
    if not isinstance(samples,list) or not samples or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in samples):
        raise ValueError('raw duration samples must be finite, positive milliseconds')
    ordered=sorted(samples);index=(len(ordered)-1)*.95;lo=math.floor(index);hi=math.ceil(index)
    return {'median_ms':statistics.median(samples),'p95_ms':ordered[lo]+(ordered[hi]-ordered[lo])*(index-lo)}


def build_registry(catalog,measurements):
    by={op['id']:op for op in catalog}
    if len(by)!=len(catalog) or not by:raise ValueError('nonempty unique operation catalog required')
    result=[];dependencies=dependency_identities()
    for item in measurements:
        required=('operation','variant','source_sha256','reference_sha256','workload','environment','metrics','iterations','warmups','scope','shader_tree_sha256','reference_tree_sha256')
        if not isinstance(item,dict) or any(k not in item for k in required):raise ValueError('incomplete timing identity')
        if item['operation'] not in by or item['variant']!=by[item['operation']]['variant']:raise ValueError('unregistered timing operation or variant')
        for field in ('source_sha256','reference_sha256','shader_tree_sha256','reference_tree_sha256'):
            if not isinstance(item[field],str) or not re.fullmatch('[a-f0-9]{64}',item[field]):raise ValueError('invalid source identity')
        w=item['workload'];env=item['environment']
        if not isinstance(w,dict) or not w.get('id') or not w.get('description') or not re.fullmatch('[a-f0-9]{64}',w.get('sha256','')):raise ValueError('complete workload identity required')
        if not isinstance(env,dict) or not env.get('cpu') or not item['scope']:raise ValueError('environment and measurement scope required')
        if type(item['iterations']) is not int or item['iterations']<1 or type(item['warmups']) is not int or item['warmups']<1:raise ValueError('positive iterations and warmups required')
        metrics=item['metrics']
        if not isinstance(metrics,dict) or not metrics or set(metrics)-set(METRICS):raise ValueError('unknown or absent timing metric')
        if any(k.startswith('gpu_') for k in metrics) and (not env.get('gpu') or not re.fullmatch('[a-f0-9]{64}',env.get('compiler_sha256',''))):raise ValueError('GPU and compiler identity required')
        for samples in metrics.values():
            summarize(samples)
            if len(samples)!=item['iterations']:raise ValueError('raw sample count differs from iterations')
        row=copy.deepcopy(item);row['current_source']=item['source_sha256']==by[item['operation']]['sha256'] and all(item[k]==v for k,v in dependencies.items());result.append(row)
    operations=[];cpu=gpu=paired=0
    for op in catalog:
        rows=[r for r in result if r['operation']==op['id']];current=[r for r in rows if r['current_source']]
        has_cpu=any('cpu_reference_wall_ms' in r['metrics'] for r in current);has_gpu=any('gpu_kernel_ms' in r['metrics'] for r in current)
        both=any({'cpu_reference_wall_ms','gpu_kernel_ms'}<=r['metrics'].keys() for r in current)
        cpu+=has_cpu;gpu+=has_gpu;paired+=both
        status='measured' if both else 'partial' if current else 'stale' if rows else 'pending'
        reason='' if both else 'No executable timing recipe recorded yet.' if not rows else 'Dependency/source changed; rerun this workload.' if not current else 'A matching CPU reference and GPU kernel workload remains pending.'
        operations.append({'operation':op['id'],'variant':op['variant'],'source_sha256':op['sha256'],'status':status,'reason':reason})
    return {'schema_version':1,'dependencies':dependencies,'units':'milliseconds','quantiles':'Median; p95 uses linear interpolation at (n-1)*0.95. Raw samples retained.','scope':'Explicit workload/configuration timings, not optimization or application qualification. CPU reference implementations are not necessarily optimized CPU alternatives.','coverage':{'catalog_entries':len(catalog),'cpu_measured':cpu,'gpu_measured':gpu,'both_measured':paired,'pending':len(catalog)-paired},'operations':operations,'measurements':result}


def render_markdown(database):
    lines=['# CPU and GPU kernel timing reference','',database['scope'],'',database['quantiles'],'',f"Coverage: {database['coverage']['both_measured']} / {database['coverage']['catalog_entries']} operations have a source-matched paired CPU/GPU baseline. Pending entries remain part of the required measurement work.",'','Times are median / p95 **milliseconds**. Each row is one workload and environment; pipeline costs are never copied onto individual kernels. Small correctness fixtures are not production-throughput benchmarks. Historical sources are explicitly stale. Raw samples and full identities are in [the timing registry](../src/sintergpu/data/timings.json).','','| Operation / variant | Workload | CPU reference | GPU kernel | GPU pipeline | Transfer-inclusive | Environment | State |','| --- | --- | ---: | ---: | ---: | ---: | --- | --- |']
    def safe(text):return str(text).replace('|','\\|').replace('\n',' ')
    for op in database['operations']:
        rows=[r for r in database['measurements'] if r['operation']==op['operation']] or [None]
        for row in rows:
            def cell(metric):
                if row is None or metric not in row['metrics']:return 'pending'
                if not row['current_source']:return 'stale'
                s=summarize(row['metrics'][metric]);return f"{s['median_ms']:.6f} / {s['p95_ms']:.6f}"
            environment='pending' if row is None else row['environment']['cpu']+'; '+str(row['environment'].get('gpu') or 'CPU only')+'; '+str(row['environment'].get('captured_at_utc','capture time not recorded'))
            state=op['status'] if row is None else 'current source' if row['current_source'] else 'stale source'
            workload='pending' if row is None else row['workload']['id']
            lines.append('| '+ ' | '.join(safe(v) for v in (op['operation']+' / '+op['variant'],workload,cell('cpu_reference_wall_ms'),cell('gpu_kernel_ms'),cell('gpu_pipeline_ms'),cell('transfer_inclusive_ms'),environment,state))+' |')
    if database.get('additional_pending_targets'):
        lines+=['','## Additional native variants','','These executable variants are outside the portable dispatch catalog. They also require CPU/GPU timing baselines; no build or historical upstream time substitutes for execution.','','| Variant | CPU reference | GPU kernel | State |','| --- | --- | --- | --- |']
        for row in database['additional_pending_targets']:lines.append('| '+safe(row['operation'])+' | pending | pending | '+safe(row['reason'])+' |')
    return '\n'.join(lines)+'\n'
