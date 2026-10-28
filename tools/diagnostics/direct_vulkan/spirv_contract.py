# SPDX-License-Identifier: Apache-2.0
"""Fail-closed SPIR-V ABI inspection for the two fixed point-plane modules."""
import struct


def validate(data, group):
    if len(data)<20 or len(data)%4: raise ValueError('SPIR-V size')
    words=struct.unpack('<%dI'%(len(data)//4),data)
    if words[0]!=0x07230203 or words[4]!=0: raise ValueError('SPIR-V header')
    ins=[];i=5
    while i<len(words):
        count=words[i]>>16;op=words[i]&0xffff
        if not count or i+count>len(words): raise ValueError('SPIR-V instruction')
        ins.append((op,words[i+1:i+count]));i+=count
    if not ins or ins[-1][0]!=56: raise ValueError('missing function end')
    entries=[v for op,v in ins if op==15]
    if len(entries)!=1 or entries[0][:4]!=(5,entries[0][1],0x6e69616d,0): raise ValueError('compute main entry')
    entry=entries[0][1]
    modes=[v for op,v in ins if op==16 and len(v)>=2 and v[0]==entry and v[1]==17]
    if modes!=[(entry,17,group,1,1)]: raise ValueError('workgroup')
    decorations={};member={};types={};variables={}
    for op,v in ins:
        if op==71:
            if len(v)<2: raise ValueError('decoration')
            key=(v[0],v[1]);
            if key in decorations: raise ValueError('duplicate decoration')
            decorations[key]=v[2:]
        elif op==72:
            if len(v)<3: raise ValueError('member decoration')
            key=(v[0],v[1],v[2]);
            if key in member: raise ValueError('duplicate member decoration')
            member[key]=v[3:]
        elif op in (21,22,29,30,32):
            if not v or v[0] in types: raise ValueError('duplicate type')
            types[v[0]]=(op,v[1:])
        elif op==59:
            if len(v)<3 or v[1] in variables: raise ValueError('variable')
            variables[v[1]]=(v[0],v[2])
    bindings={}
    for (id,dec),value in decorations.items():
        if dec!=33: continue
        if len(value)!=1 or decorations.get((id,34))!=(0,) or id not in variables: raise ValueError('binding/set')
        b=value[0]
        if b in bindings: raise ValueError('duplicate binding')
        bindings[b]=id
    if set(bindings)!=set(range(7)): raise ValueError('descriptor bindings')
    if any((id,33) not in decorations for id,dec in decorations if dec==34): raise ValueError('unbound set')
    for b,id in bindings.items():
        pointer,storage=variables[id]
        ptr=types.get(pointer)
        if not ptr or ptr[0]!=32 or len(ptr[1])!=2 or ptr[1][0]!=storage: raise ValueError('pointer')
        struct_id=ptr[1][1];kind,fields=types.get(struct_id,(None,()))
        if kind!=30 or (struct_id,2) not in decorations: raise ValueError('block struct')
        if b==6:
            if storage!=2 or len(fields)!=6: raise ValueError('uniform layout')
            if any(types.get(t)!=(21,(32,0)) for t in fields): raise ValueError('uniform field type')
            if any(member.get((struct_id,i,35))!=(4*i,) for i in range(6)): raise ValueError('uniform field offset')
        else:
            if storage!=12 or len(fields)!=1 or member.get((struct_id,0,35))!=(0,): raise ValueError('storage layout')
            array=fields[0];kind,values=types.get(array,(None,()))
            if kind!=29 or len(values)!=1 or decorations.get((array,6))!=(4,): raise ValueError('storage stride')
            expected=(21,(32,0)) if b==5 else (22,(32,))
            if types.get(values[0])!=expected: raise ValueError('storage scalar')
            readonly=(id,24) in decorations
            if readonly != (b in (0,1,2,5)): raise ValueError('storage access')
    return {'entry':'main','group':[group,1,1],'bindings':7,'uniform_offsets':[0,4,8,12,16,20],
            'uniform_allocated_bytes':32,'storage_stride':4}
