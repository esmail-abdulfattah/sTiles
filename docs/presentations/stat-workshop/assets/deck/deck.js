/* ══════════════════════════════════════════════════════════════════════
   The deck engine, lifted verbatim out of talk.html so a second deck can
   share it. Three parts, in the order they have to run:

     1. matrix() and the pattern generators  (drawing primitive)
     2. the injected marks                   (brand, KAUST, act)
     3. the engine                           (slides, steps, nav, fit, hash)

   A deck that includes this file must call boot() once, at the end of its
   own script, after any figure it wants drawn before the first paint.
   Hooks: window.onStep(slideIndex, step) is chained, never replaced.
   ══════════════════════════════════════════════════════════════════════ */

/* ══════════════════ matrix drawing helper ══════════════════
   One primitive for the whole talk: a spec → crisp inline SVG.
   role() returns 'a' | 'fill' | 'band' | 'border' | 'dense' | null      */
const C = {
  a:'#14161a', fill:'#c9a227',
  band:'#3c8277', border:'#dd8545',
  dense:'#dd8545', semi:'#bdcd51', sparse:'#3c8277',
  grid:'#c9cbc7', empty:'#eceee9'
};

function matrix(el, {n, cols=null, cell=13, role, tile=0, pad=1, showEmpty=true,
                     stroke=null, boxes=null, labels=false, rings=null, grid=false}){
  if(typeof el==='string') el=document.getElementById(el);
  /* fail soft: a figure whose slide was edited away must never take the
     whole deck down mid-talk */
  if(!el){ console.warn('matrix(): no such element'); return null; }
  const m=cols??n, W=m*cell, S=n*cell, NS='http://www.w3.org/2000/svg';
  const P0=labels?24:0;                        // gutter for the index labels
  const svg=document.createElementNS(NS,'svg');
  svg.setAttribute('width',W+P0); svg.setAttribute('height',S+P0);
  svg.setAttribute('viewBox',`0 0 ${W+P0} ${S+P0}`);
  const g=document.createElementNS(NS,'g');
  g.setAttribute('transform',`translate(${P0},${P0})`);
  const root=svg; svg.appendChild(g);
  /* everything below draws into g so labels can live in the gutter */
  const append=e=>g.appendChild(e);
  if(labels){
    for(let k=0;k<Math.max(n,m);k++){
      const t=(x,y)=>{const e=document.createElementNS(NS,'text');
        e.setAttribute('x',x);e.setAttribute('y',y);e.setAttribute('text-anchor','middle');
        e.setAttribute('font-size',Math.min(13,cell*0.32));
        e.setAttribute('fill','#6b7280');
        e.setAttribute('font-family','JetBrains Mono, monospace');
        e.textContent=k+1; root.appendChild(e);};
      if(k<m) t(P0+k*cell+cell/2, P0-8);
      if(k<n) t(P0-11, P0+k*cell+cell/2+4);
    }
  }
  if(showEmpty){
    const bg=document.createElementNS(NS,'rect');
    bg.setAttribute('width',W); bg.setAttribute('height',S);
    bg.setAttribute('fill',C.empty); append(bg);
  }
  for(let i=0;i<n;i++) for(let j=0;j<m;j++){
    const r=role(i,j); if(!r) continue;
    const key=Array.isArray(r)?r[0]:r, op=Array.isArray(r)?r[1]:1;
    const rect=document.createElementNS(NS,'rect');
    rect.setAttribute('x',j*cell+pad); rect.setAttribute('y',i*cell+pad);
    rect.setAttribute('width',cell-2*pad); rect.setAttribute('height',cell-2*pad);
    rect.setAttribute('fill',C[key]||key); rect.setAttribute('rx',Math.min(2,cell/6));
    if(op<1) rect.setAttribute('fill-opacity',op);
    append(rect);
  }
  /* a faint cell grid: only worth drawing when cells are big enough to count */
  if(grid){
    for(let k=0;k<=Math.max(n,m);k++){
      const ln=(x1,y1,x2,y2)=>{const l=document.createElementNS(NS,'line');
        l.setAttribute('x1',x1);l.setAttribute('y1',y1);
        l.setAttribute('x2',x2);l.setAttribute('y2',y2);
        l.setAttribute('stroke','#c9cbc7');l.setAttribute('stroke-width',1);append(l);};
      if(k<=m) ln(k*cell,0,k*cell,S);
      if(k<=n) ln(0,k*cell,W,k*cell);
    }
  }
  if(tile){
    for(let k=0;k<=n;k+=tile){
      for(const [x1,y1,x2,y2] of [[k*cell,0,k*cell,S],[0,k*cell,W,k*cell]]){
        const l=document.createElementNS(NS,'line');
        l.setAttribute('x1',x1);l.setAttribute('y1',y1);
        l.setAttribute('x2',x2);l.setAttribute('y2',y2);
        l.setAttribute('stroke','#14161a');l.setAttribute('stroke-width',1);
        l.setAttribute('opacity',.32); append(l);
      }
    }
  }
  if(stroke){
    const o=document.createElementNS(NS,'rect');
    o.setAttribute('width',W);o.setAttribute('height',S);o.setAttribute('fill','none');
    o.setAttribute('stroke',stroke);o.setAttribute('stroke-width',1.5);append(o);
  }
  for(const b of (boxes||[])){
    const o=document.createElementNS(NS,'rect');
    o.setAttribute('x',b.j0*cell); o.setAttribute('y',b.i0*cell);
    o.setAttribute('width',(b.j1-b.j0+1)*cell); o.setAttribute('height',(b.i1-b.i0+1)*cell);
    o.setAttribute('fill',b.fill||'none'); o.setAttribute('stroke',b.color||'#3c8277');
    o.setAttribute('stroke-width',b.w||2); append(o);
  }
  /* ring one cell so the spoken line can point at it */
  for(const r of (rings||[])){
    const o=document.createElementNS(NS,'rect');
    o.setAttribute('x',r.j*cell-1.5); o.setAttribute('y',r.i*cell-1.5);
    o.setAttribute('width',cell+3); o.setAttribute('height',cell+3);
    o.setAttribute('fill','none'); o.setAttribute('stroke',r.color||'#3c8277');
    o.setAttribute('stroke-width',2.5); o.setAttribute('rx',3);
    if(r.dashed) o.setAttribute('stroke-dasharray','4 3');
    append(o);
  }
  el.innerHTML=''; el.appendChild(svg); return svg;
}

/* ---------- pattern generators ---------- */
const P = {
  chain: n => (i,j) => Math.abs(i-j)<=1 ? 'a' : null,
  dense: n => (i,j) => 'dense',
  band : (n,w) => (i,j) => Math.abs(i-j)<=w ? 'band' : null,
  arrow: (n,w,b) => (i,j) =>
      (i>=n-b || j>=n-b) ? 'border' : (Math.abs(i-j)<=w ? 'band' : null),
  blockBand: (n,blk,w) => (i,j) => {
      const bi=Math.floor(i/blk), bj=Math.floor(j/blk);
      if(Math.abs(bi-bj)>1) return null;
      return (bi===bj ? Math.abs(i-j)<=w : Math.abs(i-j)%blk<=w) ? 'band' : null;
  },
  /* deterministic pseudo-random symmetric graph */
  graph: (n,deg) => {
    const h=(a,b)=>{const x=Math.sin(a*127.1+b*311.7)*43758.5453;return x-Math.floor(x);};
    return (i,j)=>{ if(i===j) return 'a';
      const [p,q]=i<j?[i,j]:[j,i];
      return h(p,q) < deg/n ? 'a' : null; };
  },
  denseBlocks: (n,blk) => (i,j) => {
    const bi=Math.floor(i/blk), bj=Math.floor(j/blk);
    if(bi===bj) return 'dense';
    return (Math.abs(i-j)%blk===0 && Math.abs(bi-bj)===1) ? 'band' : null;
  },
  domDense: (n,d) => (i,j) =>
      (i<d&&j<d) ? 'dense' : (i===j ? 'a' : (Math.abs(i-j)===7 ? 'band' : null))
};

/* the standing corner mark, injected rather than authored, so that no slide
   can be added without it and none has to carry the markup */
(function(){
  document.querySelectorAll('.slide:not(.no-mark)').forEach(s=>{
    if(s.querySelector('.brandmark')) return;
    const m=document.createElement('div'); m.className='brandmark';
    m.innerHTML='<i></i><i></i><i></i><i></i>';
    m.title='sTiles';
    s.appendChild(m);
  });
})();

/* the institutional mark on every slide, injected rather than authored for
   the same reason as the others: one place to change it, and no slide can
   be added without it. The seed reads on paper and on the dark dividers
   alike, so there is no inverted variant to pick between. */
(function(){
  document.querySelectorAll('.slide').forEach(s=>{
    if(s.querySelector('.kaust')) return;
    const a=document.createElement('img');
    a.className='kaust'; a.src='assets/brand/kaust-logo.svg'; a.alt='KAUST';
    s.appendChild(a);
  });
})();

/* the act mark, same deal: tag a slide with data-act="dense" and it carries
   the standing label for that stretch of the talk. One place to edit the
   wording, and a slide can never end up labelled in two different ways.
   An unknown value is left undrawn rather than guessed at, so a typo shows
   up as a missing mark instead of as a mislabelled act. */
(function(){
  const NAME={dense:'dense solver', sparse:'sparse solver',
              semi:'semi-structured'};
  document.querySelectorAll('.slide[data-act]').forEach(s=>{
    const a=s.dataset.act;
    if(!NAME[a] || s.querySelector('.actmark')) return;
    const m=document.createElement('div');
    m.className='actmark '+a;
    m.innerHTML='<i></i>'+NAME[a];
    s.appendChild(m);
  });
})();


/* ══════════════════ deck engine ══════════════════ */
const slides=[...document.querySelectorAll('.slide')];
let i=0, st=0, shown=-1;
const steps=s=>[...s.querySelectorAll('.step')];
/* The last beat of a slide is not always carried by a .step element.
   Ring groups are driven by data-rings through the onStep hook below, and
   counting only .step elements left the final ring beat unreachable: the
   deck would simply advance to the next slide instead of showing it. */
const maxStep=s=>{
  let m=steps(s).reduce((m,e)=>Math.max(m,+e.dataset.step||0),0);
  if(s.dataset.rings)
    for(const p of s.dataset.rings.trim().split(/\s+/))
      m=Math.max(m, +p.split(':')[1]||0);
  return m;
};

function draw(){
  /* Only on a real slide change, not on every step within one. */
  if(shown!==i){
    const old=slides[shown];
    slides.forEach(s=>s.classList.remove('out'));
    if(old && old!==slides[i]){
      old.classList.add('out');
      const clear=()=>old.classList.remove('out');
      old.addEventListener('animationend',clear,{once:true});
      setTimeout(clear,600);        /* in case the animation never fires */
    }
    const cur=slides[i];
    cur.classList.add('no-anim');    /* steps below land without sliding */
    requestAnimationFrame(()=>requestAnimationFrame(
      ()=>cur.classList.remove('no-anim')));
    shown=i;
  }
  slides[i].classList.remove('out'); /* navigating back mid-fade */
  slides.forEach((s,k)=>s.classList.toggle('on',k===i));
  const s=slides[i];
  steps(s).forEach(e=>e.classList.toggle('vis',(+e.dataset.step||0)<=st));
  /* elements that belong to the opening beat and step aside afterwards */
  s.querySelectorAll('[data-until]').forEach(e=>
    e.classList.toggle('gone', st > +e.dataset.until));
  if(window.onStep) window.onStep(i, st);
  paintNo(s);
  document.getElementById('bar').style.width=((i+1)/slides.length*100)+'%';
}
/* ══════ the slide number appears only when reached for ══════
   Distance from the pointer to the active slide's number box, in screen
   pixels. #stage is transform-scaled, but getBoundingClientRect reports
   post-transform coordinates, so this stays correct at any window size. */
(function(){
  const NEAR=170;
  let on=false;
  addEventListener('mousemove',function(e){
    const box=document.querySelector('.slide.on .slide-no');
    if(!box) return;
    const r=box.getBoundingClientRect();
    const dx=Math.max(r.left-e.clientX, 0, e.clientX-r.right);
    const dy=Math.max(r.top-e.clientY, 0, e.clientY-r.bottom);
    const near=Math.hypot(dx,dy)<NEAR;
    if(near!==on){ on=near; document.documentElement.classList.toggle('near-no',near); }
  },{passive:true});
  /* a pointer that leaves the window cannot be near anything */
  addEventListener('mouseleave',function(){
    on=false; document.documentElement.classList.remove('near-no');
  });
})();

function go(d){
  const s=slides[i], m=maxStep(s);
  if(d>0){ if(st<m){st++;} else if(i<slides.length-1){i++;st=0;} }
  else   { if(st>0){st--;} else if(i>0){i--;st=maxStep(slides[i]);} }
  draw();
}
/* ── the counter is the way in ───────────────────────────────────────
   The "4 / 5" in the corner is a field. Click it and type, or just start
   typing a digit anywhere and it takes the focus itself: 12 ⏎ lands on
   slide 12 fully revealed, the same as the #12 hash. "12.3" lands with 3
   steps shown, so a build can be entered part-way. Esc or clicking away
   puts the real number back. */

/* a human label per slide. The deck already names every section in the comment
   above it ("<!-- ══ 3 · THREE EQUATIONS ══ -->"), so read that; fall back to a
   heading for any slide written without one. */
const slideLabel=sl=>{
  for(let n=sl.previousSibling;n;n=n.previousSibling){
    /* a slide may carry its own <style> between its comment and itself;
       those render nothing, so walk past them. Any OTHER element is the
       previous slide, and there is no comment of ours left to find. */
    if(n.nodeType===1){
      if(n.tagName==='STYLE'||n.tagName==='SCRIPT') continue;
      break;
    }
    if(n.nodeType===8){
      /* the FIRST LINE only: a comment with an essay under its header
         would otherwise put the title, then prose, then an ellipsis in
         the jump box. */
      const t=n.textContent.split('\n')[0]
               .replace(/═+/g,' ').replace(/^\s*\d+\s*·\s*/,'')
               .replace(/\s+/g,' ').trim();
      if(t) return t;
      break;
    }
  }
  const h=sl.querySelector('h1,h2,.statement .big')||sl.querySelector('.kicker');
  return (h?.textContent||'').replace(/\s+/g,' ').trim();
};

/* every slide carries its own .slide-no; give each the same three parts once */
slides.forEach(sl=>{
  const box=sl.querySelector('.slide-no');
  if(!box) return;
  box.title='Type a slide number';
  box.innerHTML='<span class="peek"></span>'
               +'<input class="num" type="text" inputmode="numeric" '
               +'autocomplete="off" spellcheck="false" aria-label="go to slide">'
               +'<span class="tot"></span>';
});
const noBox=()=>slides[i].querySelector('.slide-no');
const noNum=()=>noBox()?.querySelector('.num');

function paintNo(sl){
  const box=sl.querySelector('.slide-no'); if(!box) return;
  box.classList.remove('editing','bad');
  box.querySelector('.num').value=i+1;
  box.querySelector('.peek').textContent='';
  box.querySelector('.tot').textContent='/ '+slides.length;
  sizeNo(box);
}
function sizeNo(box){
  const el=box.querySelector('.num');
  el.style.width=Math.max(1,el.value.length)+1.2+'ch';   /* room for the caret */
}
/* read what is typed so far: valid target, and what it is called */
function readNo(box){
  const m=/^(\d+)(?:\.(\d*))?$/.exec(box.querySelector('.num').value.trim());
  if(!m) return null;
  const n=+m[1];
  if(n<1||n>slides.length) return null;
  return {idx:n-1, step:m[2]};
}
function peekNo(){
  const box=noBox(), el=box.querySelector('.num'), raw=el.value.trim();
  const t=readNo(box);
  box.classList.toggle('bad', raw!=='' && !t);
  box.querySelector('.peek').textContent =
      t ? slideLabel(slides[t.idx])+(t.step ? ' · step '+t.step : '')
        : (raw==='' ? '' : 'no such slide');
  sizeNo(box);
}
function commitNo(){
  const box=noBox(), t=readNo(box);
  if(!t){ paintNo(slides[i]); box.querySelector('.num').blur(); return; }
  i=t.idx;
  st=t.step!==undefined ? Math.min(+t.step||0,maxStep(slides[i])) : maxStep(slides[i]);
  /* keep the URL honest, so a reload stays on the slide just reached */
  history.replaceState(null,'',`#${i+1}.${st}`);
  box.querySelector('.num').blur();
  draw();
}
/* the field's own keys never reach the deck: digits are digits in here */
addEventListener('keydown',e=>{
  if(!e.target.classList?.contains('num')) return;
  e.stopPropagation();
  if(e.key==='Enter'){ e.preventDefault(); commitNo(); }
  else if(e.key==='Escape'){ e.preventDefault(); paintNo(slides[i]); e.target.blur(); }
},true);
addEventListener('input',e=>{
  if(!e.target.classList?.contains('num')) return;
  e.target.value=e.target.value.replace(/[^\d.]/g,'').slice(0,7);
  peekNo();
});
addEventListener('focusin',e=>{
  if(!e.target.classList?.contains('num')) return;
  noBox().classList.add('editing');
  e.target.select();
});
addEventListener('focusout',e=>{
  if(!e.target.classList?.contains('num')) return;
  paintNo(slides[i]);                      /* nothing committed: put it back */
});

addEventListener('keydown',e=>{
  if(e.ctrlKey||e.metaKey||e.altKey) return;        /* leave browser keys alone */
  /* a digit anywhere starts a jump: the counter takes the focus and the key */
  if(/^\d$/.test(e.key)){
    e.preventDefault();
    const el=noNum(); if(!el) return;
    el.focus(); el.value=e.key; peekNo();
    return;
  }
  if(['ArrowRight',' ','PageDown','n'].includes(e.key)){e.preventDefault();go(1);}
  else if(['ArrowLeft','PageUp','p'].includes(e.key)){e.preventDefault();go(-1);}
  else if(e.key==='f'){document.documentElement.requestFullscreen?.();}
  else if(e.key==='Home'){i=0;st=0;draw();}
  else if(e.key==='End'){i=slides.length-1;st=maxStep(slides[i]);draw();}
});
/* ── click to advance, except when the click is really a read ──────────
   A drag, a double or triple click, and a click while something is still
   selected are all attempts to select or copy text, not to turn the page.
   Without this, selecting a formula advances the deck instead.         */
let dnX=0, dnY=0;
addEventListener('mousedown',e=>{ dnX=e.clientX; dnY=e.clientY; });
addEventListener('click',e=>{
  if(e.target.closest('.no-advance,.slide-no')) return;
  if(e.detail>1) return;                                    /* double, triple */
  if(Math.hypot(e.clientX-dnX,e.clientY-dnY)>6) return;     /* a drag */
  const sel=window.getSelection?.();
  if(sel && !sel.isCollapsed && sel.toString().trim()) return;
  go(e.clientX<innerWidth*0.25?-1:1);
});

function fit(){
  const s=Math.min(innerWidth/1280, innerHeight/720);
  document.getElementById('stage').style.transform=`scale(${s})`;
}
/* #7 jumps to slide 7; #7.2 jumps to slide 7 with 2 steps revealed */
function fromHash(){
  const m=/^#(\d+)(?:\.(\d+))?$/.exec(location.hash);
  if(!m) return false;
  i=Math.min(Math.max(+m[1]-1,0),slides.length-1);
  st=m[2]!==undefined ? +m[2] : maxStep(slides[i]);
  return true;
}
addEventListener('hashchange',()=>{ if(fromHash()) draw(); });
addEventListener('resize',()=>fit());

/* A deck calls this once, at the very end of its own script, after anything
   it wants painted before the first frame. Kept explicit rather than run on
   load so a deck can draw its figures first and the opening slide is never
   shown half-built. */
function boot(){ fit(); fromHash(); draw(); }
