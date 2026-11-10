"""URDF 충돌 형상(visual 메시)으로 hover→grasp 사이 책 구역 위 최저 z를 계산한다. 모델 값이다. 장치 없음."""
import sys, struct, re
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(sys.argv[1]); MESH=Path(sys.argv[2])
sys.path.insert(0,str(ROOT/"src"))
from learned_grasp import kinematics as K
from learned_grasp.config import load_config
from learned_grasp.pose_table import table_for
cfg=load_config(ROOT/"config"/"workspace.yaml"); tab=table_for(cfg)
e=tab.entries[4]
hov=np.array((90,)+tuple(e.hover)+(90,),float); gra=np.array((90,)+tuple(e.grasp)+(90,),float)
root=ET.parse(ROOT/"assets"/"urdf"/"DOFBOT_Pro-V24.urdf").getroot()
def vec(s,d=(0,0,0)): return np.array([float(x) for x in s.split()]) if s else np.array(d,float)
joints={}
for j in root.findall("joint"):
    o=j.find("origin"); ax=j.find("axis")
    joints[j.get("name")]=dict(type=j.get("type"),parent=j.find("parent").get("link"),child=j.find("child").get("link"),
        xyz=vec(o.get("xyz")) if o is not None else np.zeros(3),rpy=vec(o.get("rpy")) if o is not None else np.zeros(3),
        axis=vec(ax.get("xyz"),(1,0,0)) if ax is not None else np.array([1.,0,0]))
links={}
for l in root.findall("link"):
    v=l.find("visual")
    if v is None: continue
    m=v.find("geometry/mesh"); o=v.find("origin")
    if m is None: continue
    links[l.get("name")]=dict(file=Path(m.get("filename")).name,scale=vec(m.get("scale"),(1,1,1)),
        xyz=vec(o.get("xyz")) if o is not None else np.zeros(3),rpy=vec(o.get("rpy")) if o is not None else np.zeros(3))
def stl(path):
    b=Path(path).read_bytes()
    if b[:5]==b"solid" and b"facet" in b[:300]:
        v=np.array([[float(x) for x in m.groups()] for m in re.finditer(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)",b)])
        return v
    n=struct.unpack("<I",b[80:84])[0]
    a=np.frombuffer(b[84:84+n*50],dtype=np.dtype([("n","<f4",3),("v","<f4",(3,3)),("a","<u2")]))
    return a["v"].reshape(-1,3).astype(float)
verts={n:stl(MESH/d["file"]) for n,d in links.items()}
order=[]  # 부모→자식 순
parent_joint={j["child"]:(n,j) for n,j in joints.items()}
def fk_links(sdk):
    q={"Arm1_Joint":sdk[0],"Arm2_Joint":sdk[1],"Arm3_Joint":sdk[2],"Arm4_Joint":sdk[3],"Arm5_Joint":sdk[4]}
    T={}
    def pose(link):
        if link in T: return T[link]
        if link not in parent_joint: T[link]=np.eye(4); return T[link]
        n,j=parent_joint[link]; P=pose(j["parent"])
        M=np.eye(4); M[:3,:3]=K._rpy(j["rpy"]); M[:3,3]=j["xyz"]
        R=np.eye(3)
        if j["type"]!="fixed":
            ang=np.radians(q.get(n,90.0)-90.0)
            R=K._axis_rotations(j["axis"],np.array([ang]))[0]
        Q=np.eye(4); Q[:3,:3]=R
        T[link]=P@M@Q; return T[link]
    for l in links: pose(l)
    return T
def world_vertices(sdk):
    T=fk_links(sdk); out={}
    for n,d in links.items():
        V=verts[n]*d["scale"]
        Mv=np.eye(4); Mv[:3,:3]=K._rpy(d["rpy"]); Mv[:3,3]=d["xyz"]
        W=(T[n]@Mv@np.c_[V,np.ones(len(V))].T).T[:,:3]*1000.0
        out[n]=W
    return out
# 단위 점검: base 직립 park의 최대 z
w=world_vertices(np.array([90,90,90,90,90],float)); print("단위 점검 park 최대 z mm", round(max(v[:,2].max() for v in w.values()),1))
ZONE=(-110,110,175,285)  # x, y 구역(놓는 구역 y185~275에 10 mm 여유)
print("링크별 구역(x±110, y175~285) 안 최저 z(mm), 보간 0=hover ... 1=grasp")
print("t   "+"  ".join(f"{n[:10]:>10}" for n in links))
for t in np.linspace(0,1,6):
    q=hov*(1-t)+gra*t
    w=world_vertices(q); row=[]
    for n,V in w.items():
        m=(V[:,0]>=ZONE[0])&(V[:,0]<=ZONE[1])&(V[:,1]>=ZONE[2])&(V[:,1]<=ZONE[3])
        row.append(f"{V[m,2].min():10.1f}" if m.any() else f"{'-':>10}")
    allz=[V[:,2].min() for V in w.values()]
    print(f"{t:.1f} "+"  ".join(row)+f"   전체최저z {min(allz):.1f}")
for name,q in (("hover",hov),("grasp",gra)):
    w=world_vertices(q); print(name,"각 링크 최저 z",{n[:8]:round(V[:,2].min(),1) for n,V in w.items()})
