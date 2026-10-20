import numpy as np, networkx as nx, matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch
for f in fm.findSystemFonts():
    if 'texgyretermes' in f: fm.fontManager.addfont(f)
mpl.rcParams.update({'font.family':'TeX Gyre Termes','mathtext.fontset':'stix','font.size':10,'pdf.fonttype':3})

# ---------- Figura 1: red de dos capas ----------
rng=np.random.default_rng(7)
n=16
pos=rng.uniform(0.08,0.92,(n,2))
# separar un poco los nodos
for _ in range(200):
    for i in range(n):
        d=pos[i]-pos; r=np.linalg.norm(d,axis=1); r[i]=1
        pos[i]+=0.002*np.sum(d[r<0.18]/r[r<0.18,None]**2,axis=0)*0.01
pos=np.clip(pos,0.06,0.94)
G1=nx.barabasi_albert_graph(n,2,seed=3)
G2=nx.random_regular_graph(3,n,seed=5)
x=rng.uniform(0,1,n)          # opiniones
s=rng.random(n)<0.35          # activos
def T(p,off): return np.array([p[0]*3.2+p[1]*1.3, p[1]*1.1+off])
off1,off2=2.0,0.0
fig,ax=plt.subplots(figsize=(6.5,3.9))
cmap=plt.cm.coolwarm
for off,lab in [(off1,'Capa 1: opinión (en línea)'),(off2,'Capa 2: conducta (fuera de línea)')]:
    c=[T(p,off) for p in [(0,0),(1,0),(1,1),(0,1)]]
    ax.add_patch(Polygon(c,closed=True,fc='#f2f2f2',ec='#999999',lw=0.8,zorder=0))
    ax.text(c[2][0]+0.15,c[2][1]-0.3,lab.replace(': ',':\n'),fontsize=10,va='top')
P1={i:T(pos[i],off1) for i in range(n)}; P2={i:T(pos[i],off2) for i in range(n)}
for i in range(n):
    ax.plot([P1[i][0],P2[i][0]],[P1[i][1],P2[i][1]],ls=':',c='#aaaaaa',lw=0.7,zorder=1)
for u,v in G1.edges(): ax.plot(*zip(P1[u],P1[v]),c='#555555',lw=0.8,zorder=2)
for u,v in G2.edges(): ax.plot(*zip(P2[u],P2[v]),c='#555555',lw=0.8,zorder=2)
for i in range(n):
    ax.scatter(*P1[i],s=70,c=[cmap(x[i])],ec='k',lw=0.6,zorder=4)
    ax.scatter(*P2[i],s=70,c='#2b2b2b' if s[i] else 'white',ec='k',lw=0.6,zorder=4)
# medio de la campaña
M=np.array([-0.55,3.65])
for i in range(n):
    ax.add_patch(FancyArrowPatch(M,P1[i],arrowstyle='-|>',mutation_scale=6,color='#e08a00',lw=0.5,alpha=0.55,shrinkA=9,shrinkB=4,zorder=3))
ax.scatter(*M,s=260,marker='s',c='#e08a00',ec='k',lw=0.8,zorder=5)
ax.text(M[0],M[1]+0.25,'Campaña ($x_m$ fija)',ha='center',fontsize=10)
# leyenda manual
ax.scatter([],[],s=50,c='#2b2b2b',ec='k',label='activo')
ax.scatter([],[],s=50,c='white',ec='k',label='inactivo')
ax.plot([],[],ls=':',c='#aaaaaa',label='misma persona en ambas capas')
ax.plot([],[],c='#e08a00',label='influencia de la campaña')
ax.legend(loc='lower left',bbox_to_anchor=(-0.02,-0.2),ncol=4,frameon=False,fontsize=8.5,handlelength=1.5,columnspacing=1)
sm=plt.cm.ScalarMappable(cmap=cmap,norm=plt.Normalize(0,1))
cb=fig.colorbar(sm,ax=ax,fraction=0.03,pad=0.01,shrink=0.5,anchor=(0,0.9))
cb.set_label('opinión $x_i$',fontsize=9); cb.ax.tick_params(labelsize=8)
ax.set_xlim(-1.2,6.3); ax.set_ylim(-0.3,4.0); ax.axis('off')
fig.savefig('figuras/red_dos_capas.png',dpi=300,bbox_inches='tight')

# ---------- helpers de diagramas ----------
def box(ax,xy,w,h,txt,fc='#f2f2f2',fs=9.5,ec='#555555'):
    ax.add_patch(FancyBboxPatch((xy[0]-w/2,xy[1]-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.06',fc=fc,ec=ec,lw=0.8))
    ax.text(xy[0],xy[1],txt,ha='center',va='center',fontsize=fs,linespacing=1.15)
def arr(ax,a,b,txt=None,side='left',rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,color='#333333',lw=0.8,connectionstyle=f'arc3,rad={rad}'))
    if txt:
        mx,my=(a[0]+b[0])/2,(a[1]+b[1])/2
        ax.text(mx+(0.08 if side=='right' else -0.08),my,txt,fontsize=8.5,ha='left' if side=='right' else 'right',va='center',style='italic')

# ---------- Figura 2: un paso de la simulación ----------
fig,ax=plt.subplots(figsize=(6.3,4.6))
W,H=3.2,0.58
X=1.9
ys=[4.4,3.6,2.8,2.0,1.2,0.4]
txt=['Se elige una persona $i$ al azar en la capa 1\n(estos 4 cuadros se repiten $N$ veces)',
     'El algoritmo le muestra un vecino $j$:\nmás probable si piensa parecido ($\\gamma$)',
     '¿Sus opiniones están a menos de $\\epsilon$?',
     'Con probabilidad $p_m$, $i$ ve la campaña;\nsi está a menos de su umbral, se le acerca',
     'Se actualiza la capa 2: cada persona puede\nactivarse o dejar de estar activa',
     '¿Ya no cambia nada?']
fcs=['#f2f2f2','#f2f2f2','#fff4d6','#fde3c3','#e6e6e6','#fff4d6']
for y,t,f in zip(ys,txt,fcs): box(ax,(X,y),W,H,t,fc=f,fs=9)
for a,b in zip(ys[:-1],ys[1:]): arr(ax,(X,a-H/2),(X,b+H/2))
box(ax,(4.9,2.8),1.9,H,'Ambos promedian\nsus opiniones',fc='#dfeedd')
arr(ax,(X+W/2,2.8),(4.9-0.95,2.8)); ax.text(3.73,2.88,'sí',fontsize=8.5,style='italic',ha='center')
arr(ax,(4.9,2.8-H/2),(X+W/2,2.0+0.1),rad=-0.15)
ax.text(X-0.1,2.4,'no',fontsize=8.5,style='italic',ha='right')
box(ax,(4.9,0.4),1.9,H,'Fin: se guardan\nlas métricas',fc='#dfeedd')
arr(ax,(X+W/2,0.4),(4.9-0.95,0.4)); ax.text(3.73,0.48,'sí',fontsize=8.5,style='italic',ha='center')
ax.add_patch(FancyArrowPatch((X-W/2,0.4),(X-W/2,4.4),arrowstyle='-|>',mutation_scale=9,color='#333333',lw=0.8,connectionstyle='bar,fraction=-0.12'))
ax.text(X-W/2-0.95,2.4,'no:\nsiguiente\npaso',fontsize=8.5,style='italic',ha='center',va='center')
ax.set_xlim(-1.25,5.9); ax.set_ylim(0,4.8); ax.axis('off')
fig.savefig('figuras/paso_simulacion.png',dpi=300,bbox_inches='tight')

# ---------- Figura 3: flujo general del trabajo ----------
fig,ax=plt.subplots(figsize=(6.5,1.55))
labs=['Validación\ncampo medio,\n$N=100$','Red de\ndos capas,\n$N=1000$','Barrido de\nparámetros','Etiquetar el\nefecto de\ncada campaña','Predictor\n(aprendizaje\nautomático)']
xs=np.linspace(0.65,5.85,5)
for i,(xx,l) in enumerate(zip(xs,labs)):
    box(ax,(xx,0.55),1.12,0.95,l,fc=['#f2f2f2','#f2f2f2','#fde3c3','#fff4d6','#dfeedd'][i],fs=8.3)
    ax.text(xx,1.12,f'Objetivo {i+2}' if i<2 else ['Objetivo 4','Objetivo 5','Objetivo 5'][i-2],ha='center',fontsize=8,color='#666666')
for a,b in zip(xs[:-1],xs[1:]): arr(ax,(a+0.58,0.55),(b-0.58,0.55))
ax.set_xlim(0,6.5); ax.set_ylim(0.05,1.25); ax.axis('off')
fig.savefig('figuras/flujo_trabajo.png',dpi=300,bbox_inches='tight')
