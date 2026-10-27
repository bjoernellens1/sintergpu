"""Independent reference, recipe, and opt-in physical-device primitive checks."""
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np

from sintergpu.primitives import scan_layout, record_scan
from sintergpu.references.scan import segmented_scan, stable_compact

ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT/'src/sintergpu/kernels/primitives'


class PrimitiveCPUTests(unittest.TestCase):
    def test_nerfacc_packed_example_all_four_operations(self):
        data=np.arange(1,10,dtype=np.float32)
        ids=np.repeat([0,2,3],[2,3,4]) # ID1 is an empty packed segment.
        np.testing.assert_array_equal(segmented_scan(data,ids),[1,3,3,7,12,6,13,21,30])
        np.testing.assert_array_equal(segmented_scan(data,ids,exclusive=True),[0,1,0,3,7,0,6,13,21])
        np.testing.assert_array_equal(segmented_scan(data,ids,product=True),[1,2,3,12,60,6,42,336,3024])
        np.testing.assert_array_equal(segmented_scan(data,ids,product=True,exclusive=True),[1,1,1,3,12,1,6,42,336])

    def test_zero_product_empty_batches_and_stability(self):
        np.testing.assert_array_equal(segmented_scan(np.array([[2,0,3],[0,4,5]],np.float32),product=True,exclusive=True),[[1,2,0],[1,0,0]])
        for shape in [(0,), (3,0), (0,1000)]:
            self.assertEqual(segmented_scan(np.empty(shape,np.float32)).shape,shape)
        np.testing.assert_array_equal(stable_compact(np.array([9,7,9,1]),[1,0,1,1]),[9,9,1])
        self.assertEqual(stable_compact(np.array([],np.uint32),[]).size,0)
        with self.assertRaises(ValueError): segmented_scan(np.ones(3),[1,0,1])

    def test_hierarchy_has_no_256_limit_and_no_uninitialized_reads(self):
        for n in [0,1,255,256,257,5000,65537,2**24+1,0xffffffff]:
            sizes,passes=scan_layout(n,exclusive=True)
            available={'values','segment_ids'}
            for step in passes:
                reads={'scanBlocks':('values','segmentIds'), 'scanCarry':('segmentIds','output','blockIds','scannedBlocks'), 'scanExclusive':('segmentIds','inclusiveValues')}[step.entry]
                self.assertTrue(all(step.bindings[k] in available for k in reads))
                self.assertTrue(all(sizes[v]>0 for v in step.bindings.values()))
                if step.entry=='scanBlocks':
                    available.update(step.bindings[k] for k in ('output','blockValues','blockIds'))
                else: available.add(step.bindings['output'])
            if n: self.assertIn('exclusive_output',available)
            else: self.assertEqual(passes,[])
            self.assertLessEqual(len(passes),10)
        for bad in [-1,2**32,1.5]:
            with self.assertRaises(ValueError): scan_layout(bad)

    def test_borrowed_recording_has_explicit_encoder_and_no_allocation(self):
        sizes,steps=scan_layout(1000,exclusive=True)
        buffers={label:object() for label in sizes};encoder=object();calls=[]
        class Kernel:
            def dispatch(self,**kwargs): calls.append(kwargs)
        kernels={name:Kernel() for name in ('scanBlocks','scanCarry','scanExclusive')}
        record_scan(kernels,buffers,1000,command_encoder=encoder,max_workgroups_x=65535,product=True,exclusive=True)
        self.assertEqual(len(calls),len(steps))
        recorded=len(calls)
        with self.assertRaises(ValueError):
            record_scan(kernels,buffers,1000,command_encoder=encoder,max_workgroups_x=1)
        self.assertEqual(len(calls),recorded)
        for call in calls:
            self.assertIs(call['command_encoder'],encoder)
            self.assertEqual(call['vars']['product'],1)
        self.assertTrue(any(step.count>256 for step in steps))

    def test_vrdx_exact_frozen_inventory_and_lineage(self):
        provenance=json.loads((ROOT/'provenance/vrdx.json').read_text())
        self.assertEqual(provenance['revision'],'d2f603251b1b007c94ee07dd9c5579b7492af6fe')
        for name,digest in provenance['artifacts'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)
        native=(ROOT/'third_party/vrdx/src/vk_radix_sort.cc').read_text()
        for forbidden in ('vkQueueSubmit','vkCreateDevice','vkCreateCommandPool','vkAllocateMemory'):
            self.assertNotIn(forbidden,native)
        provenance=json.loads((ROOT/'provenance/primitives.json').read_text())
        self.assertIn('Copyright (c) 2025 AMD-AIOSS',provenance['notices']['LICENSE.AMD'])
        self.assertIn('PyTorch',provenance['lineage'])


@unittest.skipUnless(os.environ.get('SINTERGPU_GPU_TESTS')=='1','GPU execution explicitly requested only')
class PrimitiveGPUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import slangpy as spy
        cls.spy=spy;cls.device=spy.Device(type=spy.DeviceType.vulkan)
        cls.kernels={}
        for scalar in ('float','uint'):
            source=('#define SINTERGPU_SCAN_UINT 1\n' if scalar=='uint' else '')+(SHADERS/'scan.slang').read_text()
            module=cls.device.load_module_from_source('primitive_scan_'+scalar,source)
            cls.kernels[scalar]={e:cls.device.create_compute_kernel(cls.device.link_program([module],[module.entry_point(e)])) for e in ('scanBlocks','scanCarry','scanExclusive')}
        module=cls.device.load_module_from_source('primitive_compact',(SHADERS/'compact.slang').read_text())
        cls.compact={e:cls.device.create_compute_kernel(cls.device.link_program([module],[module.entry_point(e)])) for e in ('compactPrepare','compactScatter')}

    def buffer(self,data):
        data=np.asarray(data)
        return self.device.create_buffer(element_count=max(data.size,1),struct_size=4,usage=self.spy.BufferUsage.shader_resource|self.spy.BufferUsage.unordered_access,data=data if data.size else np.zeros(1,np.uint32))

    def execute_scan(self,values,ids,product=False,exclusive=False,encoder=None):
        count=values.size;sizes,_=scan_layout(count,exclusive=exclusive)
        buffers={label:self.buffer(np.zeros(size,np.uint32)) for label,size in sizes.items()}
        buffers['values']=self.buffer(values);buffers['segment_ids']=self.buffer(ids)
        own=encoder is None
        encoder=self.device.create_command_encoder() if own else encoder
        record_scan(self.kernels['uint' if values.dtype==np.uint32 else 'float'],buffers,count,command_encoder=encoder,max_workgroups_x=int(self.device.info.limits.max_compute_dispatch_thread_groups[0]),product=product,exclusive=exclusive)
        if own:
            self.device.submit_command_buffer(encoder.finish());self.device.wait_for_idle()
        return buffers['exclusive_output' if exclusive else 'output'],buffers

    def test_segmented_all_four_operations_irregular_hierarchical_and_batches(self):
        for n in [0,1,255,256,257,5000,65537]:
            ids=(np.arange(n,dtype=np.uint32)//1000)*2
            # Additional boundaries inside a block and immediately at a block edge.
            if n: ids += (np.arange(n,dtype=np.uint32)>=17)+(np.arange(n,dtype=np.uint32)>=256)
            for product in (False,True):
                rng=np.random.default_rng(42)
                values=(rng.uniform(.999,1.001,n) if product else rng.uniform(-1,1,n)).astype(np.float32)
                if product and n>512: values[[127,512]]=0
                for exclusive in (False,True):
                    with self.subTest(n=n,product=product,exclusive=exclusive):
                        output,_=self.execute_scan(values,ids,product,exclusive)
                        actual=output.to_numpy().view(np.float32)[:n]
                        expected=segmented_scan(values,ids,product=product,exclusive=exclusive)
                        np.testing.assert_allclose(actual,expected,rtol=2e-4,atol=2e-4)

    def test_single_segment_crosses_multiple_hierarchy_levels_without_zero_reset(self):
        n=70001
        ids=np.zeros(n,np.uint32)
        for product in (False,True):
            values=np.ones(n,np.float32)
            if product: values[::256]=1.001
            for exclusive in (False,True):
                output,_=self.execute_scan(values,ids,product,exclusive)
                np.testing.assert_allclose(output.to_numpy().view(np.float32)[:n],segmented_scan(values,ids,product=product,exclusive=exclusive),rtol=2e-4,atol=2e-4)

    def test_uint_exact_and_stable_compaction_empty_sparse_dense(self):
        for n in [0,1,255,256,257,5000,65537]:
            values=np.arange(n,dtype=np.uint32)%3
            output,_=self.execute_scan(values,np.zeros(n,np.uint32),exclusive=True)
            np.testing.assert_array_equal(output.to_numpy().view(np.uint32)[:n],segmented_scan(values,np.zeros(n,np.uint32),exclusive=True))
            for mode in ('none','all','sparse'):
                with self.subTest(n=n,mode=mode):
                    predicate=np.zeros(n,np.uint32) if mode=='none' else np.ones(n,np.uint32) if mode=='all' else (np.arange(n)%7==0).astype(np.uint32)
                    payload=np.arange(n,dtype=np.uint32)[::-1].copy()
                    predicate_buffer=self.buffer(predicate);payload_buffer=self.buffer(payload)
                    flags=self.buffer(np.zeros(n,np.uint32));ids=self.buffer(np.zeros(n,np.uint32))
                    output=self.buffer(np.zeros(n,np.uint32));selected=self.buffer(np.array([0xdeadbeef],np.uint32))
                    encoder=self.device.create_command_encoder()
                    self.compact['compactPrepare'].dispatch(thread_count=[max(256,((n+255)//256)*256),1,1],vars=dict(count=n,predicate=predicate_buffer,flags=flags,segmentIds=ids,selectedCount=selected),command_encoder=encoder)
                    sizes,_=scan_layout(n)
                    scratch={label:self.buffer(np.zeros(size,np.uint32)) for label,size in sizes.items()}
                    scratch.update(values=flags,segment_ids=ids)
                    record_scan(self.kernels['uint'],scratch,n,command_encoder=encoder,max_workgroups_x=int(self.device.info.limits.max_compute_dispatch_thread_groups[0]))
                    if n:
                        self.compact['compactScatter'].dispatch(thread_count=[((n+255)//256)*256,1,1],vars=dict(count=n,predicate=predicate_buffer,payload=payload_buffer,prefix=scratch['output'],compacted=output,selectedCount=selected),command_encoder=encoder)
                    self.device.submit_command_buffer(encoder.finish());self.device.wait_for_idle()
                    expected=stable_compact(payload,predicate)
                    self.assertEqual(selected.to_numpy().view(np.uint32)[0],expected.size)
                    np.testing.assert_array_equal(output.to_numpy().view(np.uint32)[:expected.size],expected)
